"""
Utilidades para exportar datos espectrales en formato JSON listo para plotear.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
from fastapi import HTTPException

from src.atmosphere import get_gas_manager

logger = logging.getLogger(__name__)


def load_spd_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee un archivo de espectro (.txt, .spd, .tbs) y devuelve wavelengths y values.
    Soporta formatos de 2 columnas (λ, v) o una sola fila si data.ndim == 1.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # Intentar cargar con numpy
    try:
        data = np.loadtxt(path)
    except Exception as e:
        # Algunos archivos pueden tener encabezados o formatos raros, intentar saltar líneas
        try:
            data = np.loadtxt(path, skiprows=1)
        except:
            raise ValueError(f"No se pudo parsear el archivo espectral: {e}")

    if data.ndim == 1:
        if len(data) < 2:
            raise ValueError("El archivo debe tener al menos dos valores")
        # Asumir que son pares [w1, v1, w2, v2...] o [w, v]
        data = data.reshape(-1, 2)
    
    wavelengths = data[:, 0]
    values = data[:, 1]

    # Convertir µm a nm si es necesario (típicamente vienen en µm si los valores son < 100)
    if np.any(wavelengths < 100):
        wavelengths = wavelengths * 1000.0
    
    return wavelengths, values


def load_emissivity_spectrum(
    emissivity_file: str,
    wavelength_min_nm: Optional[float] = None,
    wavelength_max_nm: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga espectro de emisividad desde archivo.
    
    Args:
        emissivity_file: Ruta al archivo de emisividad
        wavelength_min_nm: Longitud de onda mínima en nm (opcional)
        wavelength_max_nm: Longitud de onda máxima en nm (opcional)
        
    Returns:
        (wavelengths_nm, emissivity_values)
    """
    try:
        wavelengths, emissivity = load_spd_file(emissivity_file)
        
        # Filtrar por rango de longitud de onda si se especifica
        if wavelength_min_nm is not None or wavelength_max_nm is not None:
            mask = np.ones(len(wavelengths), dtype=bool)
            
            if wavelength_min_nm is not None:
                mask &= (wavelengths >= wavelength_min_nm)
            
            if wavelength_max_nm is not None:
                mask &= (wavelengths <= wavelength_max_nm)
            
            wavelengths = wavelengths[mask]
            emissivity = emissivity[mask]
        
        return wavelengths, emissivity
        
    except Exception as e:
        logger.error(f"Error cargando emisividad desde {emissivity_file}: {e}")
        raise HTTPException(status_code=400, detail=f"Error cargando emisividad: {e}")


def load_reflectance_spectrum(
    reflectance_file: str,
    wavelength_min_nm: Optional[float] = None,
    wavelength_max_nm: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga espectro de reflectancia desde archivo.
    
    Args:
        reflectance_file: Ruta al archivo de reflectancia
        wavelength_min_nm: Longitud de onda mínima en nm (opcional)
        wavelength_max_nm: Longitud de onda máxima en nm (opcional)
        
    Returns:
        (wavelengths_nm, reflectance_values)
    """
    try:
        wavelengths, reflectance = load_spd_file(reflectance_file)
        
        # Filtrar por rango de longitud de onda si se especifica
        if wavelength_min_nm is not None or wavelength_max_nm is not None:
            mask = np.ones(len(wavelengths), dtype=bool)
            
            if wavelength_min_nm is not None:
                mask &= (wavelengths >= wavelength_min_nm)
            
            if wavelength_max_nm is not None:
                mask &= (wavelengths <= wavelength_max_nm)
            
            wavelengths = wavelengths[mask]
            reflectance = reflectance[mask]
        
        return wavelengths, reflectance
        
    except Exception as e:
        logger.error(f"Error cargando reflectancia desde {reflectance_file}: {e}")
        raise HTTPException(status_code=400, detail=f"Error cargando reflectancia: {e}")


def get_atmospheric_spectrum(
    gas: str = "air",
    wavelength_min_nm: Optional[int] = None,
    wavelength_max_nm: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Obtiene espectro de atenuación/transmitancia atmosférica.
    
    Args:
        gas: Tipo de gas ("air", "H2O", "CO2", "CH4", "O3")
        wavelength_min_nm: Longitud de onda mínima en nm (opcional)
        wavelength_max_nm: Longitud de onda máxima en nm (opcional)
        
    Returns:
        (wavelengths_nm, attenuation, transmittance_optional)
    """
    try:
        # Cargar datos del gas específico
        wavelengths_nm, sigma_t = get_gas_manager().load_gas(gas)
        attenuation = sigma_t  # Ya está en unidades adecuadas (neper)
        
        # Calcular transmitancia como exp(-sigma_t)
        # Asumiendo 1 km de distancia atmosférica estándar
        transmittance = np.exp(-sigma_t) * 100.0
        
        # Filtrar por rango de longitud de onda si se especifica
        if wavelength_min_nm is not None or wavelength_max_nm is not None:
            mask = np.ones(len(wavelengths_nm), dtype=bool)
            
            if wavelength_min_nm is not None:
                mask &= (wavelengths_nm >= wavelength_min_nm)
            
            if wavelength_max_nm is not None:
                mask &= (wavelengths_nm <= wavelength_max_nm)
            
            wavelengths_nm = wavelengths_nm[mask]
            attenuation = attenuation[mask]
            transmittance = transmittance[mask]
        
        return wavelengths_nm, attenuation, transmittance
        
    except Exception as e:
        logger.error(f"Error cargando datos atmosféricos para gas '{gas}': {e}")
        raise HTTPException(status_code=400, detail=f"Error cargando datos atmosféricos: {e}")


def load_blackbody_spectrum(temperature_k: float, wavelength_min_nm: float = 8000.0, 
                           wavelength_max_nm: float = 12000.0, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula radiancia de Planck (cuerpo negro) para una temperatura.
    
    Args:
        temperature_k: Temperatura en Kelvin
        wavelength_min_nm: Longitud de onda mínima
        wavelength_max_nm: Longitud de onda máxima
        num_points: Número de puntos de muestreo
        
    Returns:
        (wavelengths_nm, radiance)
    """
    try:
        from scipy import constants
        
        # Constantes físicas
        h = constants.h  # Constante de Planck
        c = constants.c  # Velocidad de la luz
        k_b = constants.k  # Constante de Boltzmann
        
        wavelengths_m = np.linspace(wavelength_min_nm * 1e-9, 
                                    wavelength_max_nm * 1e-9, 
                                    num_points)
        
        # Ley de Planck: B(λ,T) = (2hc²/λ⁵) * 1/(exp(hc/λkT) - 1)
        numerator = 2 * h * c**2 / (wavelengths_m**5)
        denominator = np.exp(h * c / (wavelengths_m * k_b * temperature_k)) - 1
        radiance = numerator / denominator
        
        wavelengths_nm = wavelengths_m * 1e9
        
        return wavelengths_nm, radiance
        
    except Exception as e:
        logger.error(f"Error calculando cuerpo negro: {e}")
        raise HTTPException(status_code=400, detail=f"Error calculando cuerpo negro: {e}")


def interpolate_spectral_data(wavelengths_source: np.ndarray, 
                             values_source: np.ndarray,
                             wavelengths_target: np.ndarray) -> np.ndarray:
    """
    Interpola datos espectrales a longitudes de onda objetivo.
    
    Args:
        wavelengths_source: Longitudes de onda originales (nm)
        values_source: Valores espectrales originales
        wavelengths_target: Longitudes de onda objetivo (nm)
        
    Returns:
        Valores interpolados
    """
    return np.interp(wavelengths_target, wavelengths_source, values_source, 
                    left=values_source[0], right=values_source[-1])


def normalize_spectral_data(values: np.ndarray, max_value: float = 1.0) -> np.ndarray:
    """
    Normaliza datos espectrales a un rango [0, max_value].
    
    Args:
        values: Valores a normalizar
        max_value: Valor máximo después de normalizacion
        
    Returns:
        Datos normalizados
    """
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val < 1e-10:
        return np.full_like(values, max_value / 2.0)
    return ((values - min_val) / (max_val - min_val)) * max_value
