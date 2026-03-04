from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from src.api.dto.cameraDTO import CameraDTO, UpdateCameraDTO
from src.api.service.config_service import ConfigService
from fastapi import Query
from typing import Dict

from src.utils.objects.objects import ObjectUtils
from src.config import PathManager, ASSETS_DIR
from src.utils.cache import get_cache_manager
import os
import shutil


config_router = APIRouter(
    prefix="/config",
    tags=["config"]
)

config_service = ConfigService()
object_utils = ObjectUtils()
path_manager = PathManager

@config_router.get("/camera")
async def get_config() -> CameraDTO:

    camera_dto = config_service.get_camera_config()

    return camera_dto

@config_router.put("/camera")
async def update_config(camera_update: UpdateCameraDTO) -> CameraDTO:

    updated_camera = config_service.update_camera_config(camera_update)

    return updated_camera

@config_router.put("/wavelengths")
async def update_wavelengths(
    wavelength_min: int = Query(..., ge=0, description="Longitud de onda minima en μm"),
    wavelength_max: int = Query(..., gt=0, description="Longitud de onda maxima en μm"),
    bands: int = Query(..., gt=2, description="Numero de bandas"),
) -> CameraDTO:

    updated_camera = config_service.update_wavelengths(wavelength_min*1000, wavelength_max*1000, bands)

    return updated_camera

@config_router.get("/air/temperature")
async def get_air_temperature() -> float:

    temperature = config_service.get_air_temperature()

    return temperature

@config_router.put("/air/temperature")
async def set_air_temperature(
    temperature: float = Query(..., description="Temperatura del aire en kelvin")
) -> float:

    config_service.set_air_temperature(temperature)

    return config_service.get_air_temperature()

@config_router.get("/air/attenuation")
async def get_air_attenuation() -> FileResponse:
    air_attenuation_path = path_manager.get_air_attenuation_path()
    
    # Si el archivo no existe, copiarlo desde assets
    if not os.path.exists(air_attenuation_path):
        air_default = os.path.join(ASSETS_DIR, "reference_data", "air.txt")
        if os.path.exists(air_default):
            os.makedirs(os.path.dirname(air_attenuation_path), exist_ok=True)
            shutil.copy(air_default, air_attenuation_path)
    
    object_utils.valid_exist_file(air_attenuation_path)

    return FileResponse(air_attenuation_path, filename="air.txt")

@config_router.put("/air/attenuation")
async def set_air_attenuation(file: UploadFile = File(...)) -> str:

    mensaje = config_service.set_air_attenuation(file)

    return mensaje

@config_router.get("/air/suggest/attenuation")
async def suggest_air_attenuation() -> list[str]:

    air_data = config_service.suggest_air_attenuation()

    return air_data

@config_router.put("/air/attenuation/filename")
async def set_air_attenuation_by_filename(
    file_name: str = Query(..., description="Nombre del archivo de atenuación (ej: 'air.txt')")
) -> FileResponse:

    config_service.set_air_attenuation_by_filename(file_name)
    air_attenuation_path = path_manager.get_air_attenuation_path()

    return FileResponse(air_attenuation_path, filename="air.txt")


@config_router.get("/cache/stats")
async def get_cache_stats() -> Dict:
    """
    Obtiene estadísticas del cache de emisiones.
    
    Returns:
        Dict con métricas:
        - hits: Número de aciertos en cache
        - misses: Número de fallos en cache
        - total_requests: Total de peticiones
        - hit_rate_percent: Porcentaje de aciertos
        - cache_size: Número de entradas en cache
        - max_size: Tamaño máximo del cache
    """
    cache = get_cache_manager()
    return cache.get_stats()


@config_router.post("/cache/clear")
async def clear_cache() -> Dict:
    """
    Limpia completamente el cache de emisiones.
    
    Returns:
        Dict con mensaje de confirmación
    """
    cache = get_cache_manager()
    cache.clear()
    return {"status": "success", "message": "Cache limpiado exitosamente"}
