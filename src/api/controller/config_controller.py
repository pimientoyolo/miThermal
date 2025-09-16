from fastapi import APIRouter

from src.api.dto.cameraDTO import CameraDTO
from src.api.service.config_service import ConfigService


config_router = APIRouter(
    prefix="/config",
    tags=["config"]
)

config_service = ConfigService()

@config_router.get("/camera")
async def get_config() -> CameraDTO:

    camera_dto = config_service.get_camera_config()

    return camera_dto
