from fastapi import APIRouter, File, UploadFile
from src.api.service.scene_service import SceneService
from src.api.service.render_service import RenderService
from fastapi.responses import FileResponse, StreamingResponse
from src.config import SCENE_DIR
import os

scene_service = SceneService()
render_service = RenderService()

scene_router = APIRouter(
    prefix="/scene",
    tags=["scene"]
)

scenes_directory = str(os.path.dirname(SCENE_DIR))

@scene_router.post("/prepare_thermal_scene")
async def prepare_thermal_scene() -> bool:
    """
    Copia scene.xml a scene_thermal.xml usando el servicio SceneService.
    """
    scene_service.prepare_thermal_scene(SCENE_DIR)
    return True

@scene_router.post("/prepare_depth_scene")
async def prepare_depth_scene() -> bool:
    """
    Copia scene.xml a scene_depth.xml usando el servicio SceneService.
    """
    scene_service.prepare_depth_scene(SCENE_DIR)
    return True

@scene_router.post("/load_scene")
async def load_scene(file: UploadFile = File(...)) -> FileResponse:    
    scene_path = scene_service.load_scene(file)
    image_path = render_service.render_basic_scene(scene_path)
    return StreamingResponse(open(image_path, "rb"), media_type="image/jpg", headers={"Content-Disposition": "attachment; filename=scene_basic.jpg"})

@scene_router.get("/render_scene_rgb")
async def load_scene_advanced() -> FileResponse:
    image_path = render_service.render_basic_scene(SCENE_DIR)
    return StreamingResponse(open(image_path, "rb"), media_type="image/jpg", headers={"Content-Disposition": "attachment; filename=scene_advanced.jpg"})

@scene_router.get("/render_scene_depth")
async def render_scene_depth() -> FileResponse:
    image_path = render_service.render_depth_image(scenes_directory + "/scene_depth.xml")
    return StreamingResponse(open(image_path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=depth.npy"})

@scene_router.get("/render_scene_thermal")
async def render_scene_thermal() -> FileResponse:
    image_path = render_service.render_thermal_image(scenes_directory + "/scene_thermal.xml")
    return StreamingResponse(open(image_path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=thermal.npy"})

@scene_router.get("/has_loaded_scene")
async def has_loaded_scene() -> bool:
    return scene_service.has_loaded_scene(SCENE_DIR)

