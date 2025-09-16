from fastapi import APIRouter

from src.api.dto.cameraDTO import CameraDTO, UpdateCameraDTO
from src.api.service.config_service import ConfigService
from fastapi import Query, HTTPException


config_router = APIRouter(
    prefix="/config",
    tags=["config"]
)

config_service = ConfigService()

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
    wavelength_min: int = Query(..., ge=0, description="Longitud de onda minima en nm"),
    wavelength_max: int = Query(..., gt=0, description="Longitud de onda maxima en nm"),
    bands: int = Query(..., gt=2, description="Numero de bandas"),
) -> CameraDTO:

    updated_camera = config_service.update_wavelengths(wavelength_min, wavelength_max, bands)

    return updated_camera