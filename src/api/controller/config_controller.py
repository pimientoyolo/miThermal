from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import io

from src.api.dto.cameraDTO import CameraDTO, UpdateCameraDTO
from src.api.service.config_service import ConfigService
from fastapi import Query, HTTPException

from src.mitsuba_core.object_utils import ObjectUtils
import src.config as config


config_router = APIRouter(
    prefix="/config",
    tags=["config"]
)

config_service = ConfigService()
object_utils = ObjectUtils()

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

    object_utils.valid_exist_file(config.AIR_ATTENUATION_FILE)

    return FileResponse(config.AIR_ATTENUATION_FILE, filename="air.txt")

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

    return FileResponse(config.AIR_ATTENUATION_FILE, filename="air.txt")