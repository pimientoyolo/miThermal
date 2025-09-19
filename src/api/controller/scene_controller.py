from fastapi import APIRouter, File, UploadFile
from fastapi import Query
from src.api.dto.cameraDTO import CameraDTO
from src.api.dto.suggestDTO import SuggestDTO
from src.api.service.scene_service import SceneService
from src.api.service.render_service import RenderService
from fastapi.responses import FileResponse, StreamingResponse
import src.config as config
import os
import os, tempfile, zipfile, uuid
from fastapi import HTTPException, BackgroundTasks

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
    scene_service.prepare_temperature_map()
    return FileResponse(config.IMAGE_DIR, media_type="image/png", filename="rgb.png")


@scene_router.get("/loaded")
async def has_loaded_scene() -> bool:
    return scene_service.has_loaded_scene(config.SCENE_DIR)

@scene_router.get("/miThermal")
async def get_full_scene() -> FileResponse:

    return scene_service.get_scene_mi_thermal()

@scene_router.post("/miThermal")
async def create_mi_thermal_scene(
    file: UploadFile = File(...)
) -> FileResponse:
    scene_service.upload_mi_thermal_scene(file)
    render_service.render_basic_scene()
    return FileResponse(config.IMAGE_DIR, media_type="image/png", filename="rgb.png")

@scene_router.get("/suggest")
async def suggest_mi_thermal_scene() -> list[str]:  
    return scene_service.get_suggested_mi_thermal_scene()

@scene_router.post("/select/default")
async def set_default_scene(
    file_name: str = Query(..., description="archivo de la escena sugerida (ej: 'room.zip')") # type: ignore
) -> FileResponse:
    scene_service.set_default_scene(file_name)
    render_service.render_basic_scene()
    return FileResponse(config.IMAGE_DIR, media_type="image/png", filename="rgb.png")