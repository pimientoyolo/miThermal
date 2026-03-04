"""
Sistema estructurado de propiedades de objetos.
Reemplaza el manejo ad-hoc con dataclasses validadas.
"""

import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)


@dataclass
class SpectralProperty:
    """
    Propiedad espectral (emisividad o reflectancia).
    
    Attributes:
        wavelengths_nm: Longitudes de onda en nanómetros
        values: Valores espectrales (0-1 para emisividad/reflectancia)
        file_hash: Hash MD5 del archivo fuente (para deduplicación)
        file_path: Ruta del archivo original
    """
    wavelengths_nm: np.ndarray
    values: np.ndarray
    file_hash: str
    file_path: str
    
    def __post_init__(self):
        """Validación después de inicialización."""
        # Convertir a numpy arrays si no lo son
        self.wavelengths_nm = np.asarray(self.wavelengths_nm, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        
        # Validar longitudes
        if len(self.wavelengths_nm) != len(self.values):
            raise ValueError(
                f"Longitudes incompatibles: wavelengths={len(self.wavelengths_nm)}, "
                f"values={len(self.values)}"
            )
        
        if len(self.wavelengths_nm) == 0:
            raise ValueError("Array de wavelengths vacío")
        
        # Validar rangos
        if np.any(self.wavelengths_nm <= 0):
            raise ValueError("Wavelengths deben ser positivas")
        
        if np.any((self.values < 0) | (self.values > 1)):
            logger.warning(
                f"Valores espectrales fuera de rango [0,1]: "
                f"min={self.values.min():.3f}, max={self.values.max():.3f}"
            )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SpectralProperty':
        """
        Carga propiedad espectral desde archivo.
        
        Formato esperado: 2 columnas tab-separated
        - Columna 0: wavelength en micrómetros (µm)
        - Columna 1: valor espectral (0-1 o 0-100)
        
        Args:
            file_path: Ruta al archivo .txt o .tbs
        
        Returns:
            SpectralProperty: Propiedad validada
        
        Raises:
            HTTPException: Si el archivo no existe o tiene formato inválido
        """
        path = Path(file_path)
        
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Archivo no encontrado: {file_path}"
            )
        
        try:
            # Leer datos
            data = np.loadtxt(path)
            
            # Manejar caso de una sola fila
            if data.ndim == 1:
                if len(data) < 2:
                    raise ValueError("El archivo debe tener al menos 2 columnas")
                data = data.reshape(1, -1)
            
            if data.shape[1] < 2:
                raise ValueError("El archivo debe tener 2 columnas: wavelength, value")
            
            # Extraer columnas
            wavelengths_um = data[:, 0]
            values = data[:, 1]
            
            # Convertir wavelengths de µm a nm
            wavelengths_nm = wavelengths_um * 1000.0
            
            # Normalizar valores si están en porcentaje (0-100)
            if values.max() > 1.5:
                values = values / 100.0
            
            # Calcular hash del archivo
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            
            return cls(
                wavelengths_nm=wavelengths_nm,
                values=values,
                file_hash=file_hash,
                file_path=str(path)
            )
            
        except Exception as e:
            logger.error(f"Error cargando archivo espectral {file_path}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error al procesar archivo: {str(e)}"
            )
    
    def interpolate_to(self, target_wavelengths_nm: np.ndarray) -> np.ndarray:
        """
        Interpola valores a un nuevo grid de longitudes de onda.
        
        Args:
            target_wavelengths_nm: Nuevas longitudes de onda
        
        Returns:
            np.ndarray: Valores interpolados
        """
        return np.interp(target_wavelengths_nm, self.wavelengths_nm, self.values)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario (para JSON).
        
        Returns:
            dict: Representación serializable
        """
        return {
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "values": self.values.tolist(),
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "num_points": len(self.wavelengths_nm),
            "wavelength_range_nm": [
                float(self.wavelengths_nm.min()),
                float(self.wavelengths_nm.max())
            ],
            "value_range": [float(self.values.min()), float(self.values.max())],
        }


@dataclass
class ObjectProperties:
    """
    Propiedades térmicas y espectrales de un objeto de escena.
    
    Attributes:
        object_id: Identificador único del objeto
        temperature_k: Temperatura en Kelvin
        emissivity: Espectro de emisividad (opcional)
        reflectance: Espectro de reflectancia (opcional)
        metadata: Metadatos adicionales
    """
    object_id: str
    temperature_k: float
    emissivity: Optional[SpectralProperty] = None
    reflectance: Optional[SpectralProperty] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cache interno para radiancia calculada
    _radiance_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validación después de inicialización."""
        if self.temperature_k <= 0:
            raise ValueError(f"Temperatura debe ser positiva: {self.temperature_k}")
        
        if self.temperature_k > 5000:
            logger.warning(
                f"Temperatura muy alta para objeto '{self.object_id}': "
                f"{self.temperature_k}K"
            )
    
    def get_radiance(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """
        Calcula radiancia de cuerpo negro modulada por emisividad.
        
        L(λ, T) = ε(λ) * B(λ, T)
        
        donde B(λ, T) es la función de Planck.
        
        Args:
            wavelengths_nm: Longitudes de onda para calcular radiancia
        
        Returns:
            np.ndarray: Radiancia espectral en W/(m^3·sr)
        """
        # Verificar si hay cache válido
        cache_key = f"{self.temperature_k}_{len(wavelengths_nm)}"
        if self._radiance_cache and self._radiance_cache.get("key") == cache_key:
            return self._radiance_cache["radiance"]
        
        # Calcular radiancia de cuerpo negro (Ley de Planck)
        from src.mitsuba_core.blackbody import planck_spectral_radiance
        
        blackbody_radiance = planck_spectral_radiance(
            wavelengths_nm, self.temperature_k
        )
        
        # Modular por emisividad si está disponible
        if self.emissivity:
            emissivity_values = self.emissivity.interpolate_to(wavelengths_nm)
            radiance = blackbody_radiance * emissivity_values
        else:
            # Asumir cuerpo negro perfecto (ε = 1)
            radiance = blackbody_radiance
        
        # Cachear resultado
        self._radiance_cache = {
            "key": cache_key,
            "radiance": radiance,
            "wavelengths_nm": wavelengths_nm
        }
        
        return radiance
    
    def invalidate_cache(self):
        """Limpia cache al cambiar temperatura o emisividad."""
        self._radiance_cache = None
    
    def update_temperature(self, new_temperature_k: float):
        """
        Actualiza temperatura e invalida cache.
        
        Args:
            new_temperature_k: Nueva temperatura en Kelvin
        """
        if new_temperature_k <= 0:
            raise ValueError(f"Temperatura debe ser positiva: {new_temperature_k}")
        
        self.temperature_k = new_temperature_k
        self.invalidate_cache()
        logger.info(
            f"Temperatura de '{self.object_id}' "
            f"actualizada a {new_temperature_k}K"
        )
    
    def update_emissivity(self, emissivity_file: str):
        """
        Actualiza emisividad desde archivo e invalida cache.
        
        Args:
            emissivity_file: Ruta al nuevo archivo de emisividad
        """
        self.emissivity = SpectralProperty.from_file(emissivity_file)
        self.invalidate_cache()
        logger.info(
            f"Emisividad de '{self.object_id}' "
            f"actualizada desde {emissivity_file}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario (para JSON o DB).
        
        Returns:
            dict: Representación serializable
        """
        result = {
            "object_id": self.object_id,
            "temperature_k": self.temperature_k,
            "metadata": self.metadata,
        }
        
        if self.emissivity:
            result["emissivity"] = self.emissivity.to_dict()
        
        if self.reflectance:
            result["reflectance"] = self.reflectance.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectProperties':
        """
        Reconstruye desde diccionario.
        
        Args:
            data: Diccionario con datos del objeto
        
        Returns:
            ObjectProperties: Objeto reconstruido
        """
        emissivity = None
        if "emissivity" in data and data["emissivity"]:
            emiss_data = data["emissivity"]
            emissivity = SpectralProperty(
                wavelengths_nm=np.array(emiss_data["wavelengths_nm"]),
                values=np.array(emiss_data["values"]),
                file_hash=emiss_data["file_hash"],
                file_path=emiss_data["file_path"]
            )
        
        reflectance = None
        if "reflectance" in data and data["reflectance"]:
            refl_data = data["reflectance"]
            reflectance = SpectralProperty(
                wavelengths_nm=np.array(refl_data["wavelengths_nm"]),
                values=np.array(refl_data["values"]),
                file_hash=refl_data["file_hash"],
                file_path=refl_data["file_path"]
            )
        
        return cls(
            object_id=data["object_id"],
            temperature_k=data["temperature_k"],
            emissivity=emissivity,
            reflectance=reflectance,
            metadata=data.get("metadata", {})
        )
