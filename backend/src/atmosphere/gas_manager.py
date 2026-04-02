"""
Gestor de gases atmosféricos sin estado global.
Reemplaza el flujo anterior que sobrescribía AIR_ATTENUATION_FILE.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import HTTPException

from src.config import PathManager

logger = logging.getLogger(__name__)

# Gases atmosféricos disponibles
AVAILABLE_GASES = ["air", "H2O", "CO2", "CH4", "O3"]


class AtmosphericGasManager:
    """
    Gestiona múltiples gases atmosféricos sin modificar estado global.
    
    - Cache interno por gas
    - Thread-safe (solo lectura después de carga)
    - No escribe archivos temporales
    - Soporta mezclas de gases
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Args:
            data_dir: Directorio con archivos de referencia de gases (*.txt)
        """
        self.data_dir = Path(data_dir or PathManager.get_attenuation_dir())
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        
        if not self.data_dir.exists():
            logger.warning(
                f"Directorio de gases atmosféricos no existe: "
                f"{self.data_dir}"
            )
    
    def list_available_gases(self) -> list[str]:
        """
        Lista todos los gases disponibles en el directorio.
        
        Returns:
            Lista de nombres de gases (sin extensión .txt)
        """
        try:
            gas_files = list(self.data_dir.glob("*.txt"))
            return sorted([f.stem for f in gas_files])
        except Exception as e:
            logger.error(f"Error listando gases: {e}")
            return AVAILABLE_GASES  # Fallback a lista predefinida
    
    def load_gas(self, gas_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Carga datos de atenuación de un gas específico.
        
        Args:
            gas_name: Nombre del gas (sin extensión, ej: 'air', 'CO2')
        
        Returns:
            tuple[np.ndarray, np.ndarray]: (wavelengths_nm, sigma_t_neper)
                - wavelengths_nm: Longitudes de onda en nanómetros (ordenadas)
                - sigma_t_neper: Coeficientes de extinción en neper
        
        Raises:
            HTTPException: Si el archivo no existe o hay error de formato
        """
        # Verificar cache
        if gas_name in self._cache:
            return self._cache[gas_name]
        
        # Construir ruta del archivo
        file_path = self.data_dir / f"{gas_name}.txt"
        
        if not file_path.exists():
            available = self.list_available_gases()
            raise HTTPException(
                status_code=404,
                detail=f"Gas '{gas_name}' no encontrado. Disponibles: {available}"
            )
        
        try:
            # Cargar datos del archivo
            # Formato esperado: [wavelength_um, transmittance, attenuation_dB]
            data = np.loadtxt(file_path)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            if data.shape[1] < 3:
                raise ValueError(
                    "El archivo debe tener 3 columnas: "
                    "wavelength, transmittance, attenuation"
                )
            
            # Extraer columnas
            wavelengths_um = data[:, 0].astype(float)
            attenuation_db = data[:, 2].astype(float)
            
            # Convertir unidades
            wavelengths_nm = wavelengths_um * 1000.0  # µm → nm
            sigma_t_neper = attenuation_db * (np.log(10.0) / 10.0)  # dB → Neper
            
            # Ordenar por longitud de onda (menor a mayor)
            order = np.argsort(wavelengths_nm)
            wavelengths_sorted = wavelengths_nm[order]
            sigma_t_sorted = sigma_t_neper[order]
            
            # Cachear resultado
            result = (wavelengths_sorted, sigma_t_sorted)
            self._cache[gas_name] = result
            
            logger.info(
                f"Gas '{gas_name}' cargado: "
                f"{len(wavelengths_sorted)} puntos espectrales"
            )
            return result
            
        except Exception as e:
            logger.error(f"Error cargando gas '{gas_name}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error al procesar archivo de gas '{gas_name}': {str(e)}"
            )
    
    def get_attenuation_at_wavelength(
        self, 
        gas_name: str, 
        wavelength_nm: float
    ) -> float:
        """
        Obtiene el coeficiente de atenuación a una longitud de onda específica.
        Interpola linealmente si la longitud de onda no está en los datos.
        
        Args:
            gas_name: Nombre del gas
            wavelength_nm: Longitud de onda en nm
        
        Returns:
            float: Coeficiente de extinción en neper
        """
        wavelengths, sigma_t = self.load_gas(gas_name)
        return float(np.interp(wavelength_nm, wavelengths, sigma_t))
    
    def get_transmittance(
        self, 
        gas_name: str, 
        distance_m: float = 1.0,
        wavelength_nm: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula la transmitancia para una distancia dada.
        
        Transmitancia = exp(-sigma_t * distance)
        
        Args:
            gas_name: Nombre del gas
            distance_m: Distancia de propagación en metros
            wavelength_nm: Longitudes de onda específicas
                (None = usar todas del archivo)
        
        Returns:
            tuple[np.ndarray, np.ndarray]: (wavelengths_nm, transmittance)
        """
        wavelengths_full, sigma_t_full = self.load_gas(gas_name)
        
        if wavelength_nm is not None:
            # Interpolar a longitudes de onda específicas
            sigma_t = np.interp(wavelength_nm, wavelengths_full, sigma_t_full)
            wavelengths = wavelength_nm
        else:
            wavelengths = wavelengths_full
            sigma_t = sigma_t_full
        
        transmittance = np.exp(-sigma_t * distance_m)
        return wavelengths, transmittance
    
    def create_medium_spectrum_string(
        self, 
        gas_name: str,
        max_points: int = 100
    ) -> str:
        """
        Genera string de espectro para XML de Mitsuba.
        
        Formato: "wavelength1:value1, wavelength2:value2, ..."
        
        Args:
            gas_name: Nombre del gas
            max_points: Número máximo de puntos (subsampling si es necesario)
        
        Returns:
            str: String formateado para Mitsuba
        """
        wavelengths, sigma_t = self.load_gas(gas_name)
        
        # Subsampling si hay demasiados puntos
        if len(wavelengths) > max_points:
            indices = np.linspace(0, len(wavelengths) - 1, max_points, dtype=int)
            wavelengths = wavelengths[indices]
            sigma_t = sigma_t[indices]
        
        # Formatear como pares wavelength:value
        pairs = [f"{w:.1f}:{s:.6e}" for w, s in zip(wavelengths, sigma_t)]
        return ", ".join(pairs)
    
    def mix_gases(
        self,
        gas_concentrations: dict[str, float],
        wavelength_nm: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mezcla múltiples gases según concentraciones (fracción volumétrica).
        
        sigma_t_mix = sum(concentration_i * sigma_t_i)
        
        Args:
            gas_concentrations: {gas_name: concentration}
                Ej: {"air": 0.78, "H2O": 0.02, "CO2": 0.0004}
            wavelength_nm: Grid común de longitudes de onda
        
        Returns:
            tuple[np.ndarray, np.ndarray]: (wavelengths, sigma_t_mixed)
        """
        # Validar que las concentraciones suman ~1.0
        total_concentration = sum(gas_concentrations.values())
        if not (0.99 <= total_concentration <= 1.01):
            logger.warning(f"Concentraciones no suman 1.0: {total_concentration}")
        
        # Determinar grid común de longitudes de onda
        if wavelength_nm is None:
            # Usar el grid más fino de todos los gases
            all_wavelengths = []
            for gas_name in gas_concentrations.keys():
                wl, _ = self.load_gas(gas_name)
                all_wavelengths.append(wl)
            wavelength_nm = np.unique(np.concatenate(all_wavelengths))
            wavelength_nm.sort()
        
        # Sumar atenuaciones ponderadas
        sigma_t_mixed = np.zeros_like(wavelength_nm, dtype=float)
        
        for gas_name, concentration in gas_concentrations.items():
            wl_gas, sigma_gas = self.load_gas(gas_name)
            sigma_interp = np.interp(wavelength_nm, wl_gas, sigma_gas)
            sigma_t_mixed += concentration * sigma_interp
        
        return wavelength_nm, sigma_t_mixed
    
    def clear_cache(self):
        """Limpia el cache de gases cargados."""
        self._cache.clear()
        logger.info("Cache de gases atmosféricos limpiado")


# Instancia global (singleton) del gestor
_gas_manager_instance: Optional[AtmosphericGasManager] = None


def get_gas_manager() -> AtmosphericGasManager:
    """
    Obtiene la instancia singleton del gestor de gases.
    
    Returns:
        AtmosphericGasManager: Instancia compartida
    """
    global _gas_manager_instance
    if _gas_manager_instance is None:
        _gas_manager_instance = AtmosphericGasManager()
    return _gas_manager_instance
