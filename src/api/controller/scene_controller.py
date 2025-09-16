from fastapi import APIRouter, File, UploadFile
from src.api.dto.cameraDTO import CameraDTO
from src.api.service.scene_service import SceneService
from src.api.service.render_service import RenderService
from fastapi.responses import FileResponse, StreamingResponse
import src.config as config
import os

scene_service = SceneService()
render_service = RenderService()

scene_router = APIRouter(
    prefix="/scene",
    tags=["scene"]
)

@scene_router.post("/load")
async def load_scene(file: UploadFile = File(...)) -> FileResponse:    
    scene_service.load_scene(file)
    render_service.render_basic_scene()
    scene_service.prepare_depth_scene()
    scene_service.prepare_thermal_scene()
    scene_service.prepare_blackbody_air_scene()
    scene_service.prepare_transmittance_blackbody_air_scene()
    return FileResponse(config.IMAGE_DIR, media_type="image/png", filename="rgb.png")


@scene_router.get("/loaded")
async def has_loaded_scene() -> bool:
    return scene_service.has_loaded_scene(config.SCENE_DIR)
