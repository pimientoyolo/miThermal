"""
Controller para endpoints de datos espectrales (JSON para plotear en frontend).
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from src.api.dto.spectralDTO import (
    SpectralDataResponse, 
    AtmosphericDataResponse,
    BlackbodyDataResponse,
    ObjectSpectralDataResponse
)
from src.utils.spectral.data_export import (
    load_emissivity_spectrum,
    load_reflectance_spectrum,
    get_atmospheric_spectrum as calculate_atmospheric_spectrum,
    load_blackbody_spectrum,
    interpolate_spectral_data
)
from src.config import get_config_scene_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spectral", tags=["spectral"])


@router.get("/object/{object_id:path}", response_model=ObjectSpectralDataResponse)
async def get_object_spectral_data(
    object_id: str,
    wavelength_min_nm: Optional[float] = Query(None, description="Longitud de onda mínima (nm)"),
    wavelength_max_nm: Optional[float] = Query(None, description="Longitud de onda máxima (nm)")
):
    """
    Retorna datos espectrales unificados (emisividad y reflectancia) de un objeto.
    Busca las rutas de los archivos SPD en la configuración de la escena.
    """
    try:
        config = get_config_scene_dict()
        obj_info = config.get("objects", {}).get(object_id)
        
        if not obj_info:
            # Reintentar con el basename si el ID tiene ruta (ej: meshes/Cube.ply -> meshes/Cube.ply)
            # En realidad el ID en config suele ser la ruta completa.
            raise FileNotFoundError(f"Objeto no encontrado en configuración: {object_id}")

        emi_path = obj_info.get("emission_spd")
        refl_path = obj_info.get("reflectance_spd")

        if not emi_path or not refl_path:
            raise FileNotFoundError(f"No se encontraron rutas SPD para el objeto: {object_id}")

        # Cargar emisividad (ojo: el archivo emission_spd es radiancia, el original suele ser Cube.txt)
        # Pero podemos cargar el reflectancia y calcular emisividad = 1 - refl si es más directo,
        # o buscar el archivo .txt original en static.
        
        import os
        from src.config import OUTPUT_STATIC_DIR
        base_name = os.path.splitext(os.path.basename(object_id))[0]
        # El archivo original de emisividad que subió el usuario o el default
        orig_emi_file = f"{OUTPUT_STATIC_DIR}/{base_name}.txt"
        
        # Intentar cargar desde el archivo original de emisividad si existe
        if os.path.exists(orig_emi_file):
            wl, emi = load_emissivity_spectrum(orig_emi_file, wavelength_min_nm, wavelength_max_nm)
            refl = 1.0 - emi
        else:
            # Fallback: cargar desde el SPD de reflectancia generado por Mitsuba
            wl, refl = load_reflectance_spectrum(refl_path, wavelength_min_nm, wavelength_max_nm)
            emi = 1.0 - refl

        return ObjectSpectralDataResponse(
            wavelengths=wl.tolist(),
            emissivity=emi.tolist(),
            reflectance=refl.tolist(),
            object_id=object_id,
            label=f"Datos espectrales: {base_name}",
            title=f"Propiedades Espectrales - {base_name}"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo datos espectrales de objeto: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/atmosphere", response_model=AtmosphericDataResponse)
async def get_atmospheric_spectrum(
    gas: str = Query("air", description="Tipo de gas: air, H2O, CO2, CH4, O3"),
    wavelength_min_nm: Optional[int] = Query(None, description="Longitud de onda mínima (nm)"),
    wavelength_max_nm: Optional[int] = Query(None, description="Longitud de onda máxima (nm)")
):
    """
    Retorna datos de atenuación/transmitancia atmosférica.
    
    Args:
        gas: Tipo de gas atmosférico
        wavelength_min_nm: Longitud de onda mínima en nm (opcional)
        wavelength_max_nm: Longitud de onda máxima en nm (opcional)
        
    Returns:
        AtmosphericDataResponse con atenuación y transmitancia
    """
    try:
        wavelengths_nm, attenuation, transmittance = calculate_atmospheric_spectrum(
            gas=gas,
            wavelength_min_nm=wavelength_min_nm,
            wavelength_max_nm=wavelength_max_nm
        )
        
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
    temperature_k: float = Query(300.0, ge=100.0, le=10000.0, description="Temperatura en Kelvin"),
    wavelength_min_nm: float = Query(8000.0, description="Longitud de onda mínima (nm)"),
    wavelength_max_nm: float = Query(12000.0, description="Longitud de onda máxima (nm)"),
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
