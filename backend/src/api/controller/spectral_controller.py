"""
Controller para endpoints de datos espectrales (JSON para plotear en frontend).
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from src.api.dto.spectralDTO import (
    SpectralDataResponse, 
    AtmosphericDataResponse,
    BlackbodyDataResponse
)
from src.utils.spectral.data_export import (
    load_emissivity_spectrum,
    load_reflectance_spectrum,
    get_atmospheric_spectrum as calculate_atmospheric_spectrum,
    load_blackbody_spectrum,
    interpolate_spectral_data
)
from src.config import OUTPUT_STATIC_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spectral", tags=["spectral"])


@router.get("/emissivity/{object_id}", response_model=SpectralDataResponse)
async def get_emissivity_spectrum(object_id: str):
    """
    Retorna espectro de emisividad de un objeto para plotear en el frontend.
    
    Args:
        object_id: ID del objeto (ej: "concrete.solid")
        
    Returns:
        SpectralDataResponse con wavelengths, values, unit, label
    """
    try:
        emissivity_file = f"{OUTPUT_STATIC_DIR}/{object_id}.txt"
        wavelengths, emissivity = load_emissivity_spectrum(emissivity_file)
        
        return SpectralDataResponse(
            wavelengths=wavelengths.tolist(),
            values=emissivity.tolist(),
            unit="Emisividad (0-1)",
            label=f"Espectro de emisividad: {object_id}",
            title=f"Emisividad vs Longitud de Onda - {object_id}"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"Objeto no encontrado: {object_id}"
        )
    except Exception as e:
        logger.error(f"Error obteniendo emisividad: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reflectance/{object_id}", response_model=SpectralDataResponse)
async def get_reflectance_spectrum(object_id: str):
    """
    Retorna espectro de reflectancia de un objeto para plotear.
    
    Args:
        object_id: ID del objeto
        
    Returns:
        SpectralDataResponse con datos de reflectancia
    """
    try:
        reflectance_file = f"{OUTPUT_STATIC_DIR}/{object_id}_reflectance.txt"
        wavelengths, reflectance = load_reflectance_spectrum(reflectance_file)
        
        return SpectralDataResponse(
            wavelengths=wavelengths.tolist(),
            values=reflectance.tolist(),
            unit="Reflectancia (0-1)",
            label=f"Espectro de reflectancia: {object_id}",
            title=f"Reflectancia vs Longitud de Onda - {object_id}"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Datos de reflectancia no encontrados: {object_id}"
        )
    except Exception as e:
        logger.error(f"Error obteniendo reflectancia: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/atmosphere", response_model=AtmosphericDataResponse)
async def get_atmospheric_spectrum(
    gas: str = Query("air", description="Tipo de gas: air, H2O, CO2, CH4, O3")
):
    """
    Retorna datos de atenuación/transmitancia atmosférica.
    
    Args:
        gas: Tipo de gas atmosférico
        
    Returns:
        AtmosphericDataResponse con atenuación y transmitancia
    """
    try:
        wavelengths_nm, attenuation, transmittance = calculate_atmospheric_spectrum(gas)
        
        return AtmosphericDataResponse(
            wavelengths=wavelengths_nm.tolist(),
            attenuation=attenuation.tolist(),
            transmittance=transmittance.tolist() if transmittance is not None else None,
            gas=gas,
            label=f"Atenuación de {gas}",
            title=f"Atenuación Atmosférica - {gas}"
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo datos atmosféricos: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/blackbody", response_model=BlackbodyDataResponse)
async def get_blackbody_spectrum(
    temperature_k: int = Query(300, ge=100, le=10000, description="Temperatura en Kelvin"),
    wavelength_min_nm: int = Query(8000, description="Longitud de onda mínima (nm)"),
    wavelength_max_nm: int = Query(12000, description="Longitud de onda máxima (nm)"),
    num_points: int = Query(100, ge=10, le=1000, description="Número de puntos de muestreo")
):
    """
    Calcula y retorna espectro de radiancia de cuerpo negro (Planck).
    
    Args:
        temperature_k: Temperatura en Kelvin
        wavelength_min_nm: Longitud de onda mínima
        wavelength_max_nm: Longitud de onda máxima
        num_points: Número de puntos
        
    Returns:
        BlackbodyDataResponse con datos de Planck
    """
    try:
        wavelengths_nm, radiance = load_blackbody_spectrum(
            temperature_k, 
            wavelength_min_nm,
            wavelength_max_nm,
            num_points
        )
        
        return BlackbodyDataResponse(
            wavelengths=wavelengths_nm.tolist(),
            temperatures=[temperature_k],
            radiances=[radiance.tolist()],
            title=f"Radiancia de Planck - {temperature_k} K"
        )
        
    except Exception as e:
        logger.error(f"Error calculando cuerpo negro: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/interpolate")
async def interpolate_spectrum(
    wavelengths_source: list[float],
    values_source: list[float],
    wavelengths_target: list[float]
):
    """
    Interpola datos espectrales a nuevas longitudes de onda.
    
    Args:
        wavelengths_source: Longitudes de onda originales
        values_source: Valores originales
        wavelengths_target: Longitudes de onda objetivo
        
    Returns:
        Lista de valores interpolados
    """
    try:
        import numpy as np
        result = interpolate_spectral_data(
            np.array(wavelengths_source),
            np.array(values_source),
            np.array(wavelengths_target)
        )
        
        return {
            "wavelengths": wavelengths_target,
            "values": result.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error interpolando: {e}")
        raise HTTPException(status_code=400, detail=str(e))
