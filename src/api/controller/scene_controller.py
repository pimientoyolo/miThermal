from fastapi import APIRouter, File, UploadFile
from src.api.service.scene_service import SceneService
from src.api.service.render_service import RenderService
from fastapi.responses import FileResponse, StreamingResponse
from src.config import SCENE_DIR

scene_service = SceneService()
render_service = RenderService()

scene_router = APIRouter(
    prefix="/scene",
    tags=["scene"]
)


@scene_router.post("/prepare_thermal_scene")
async def prepare_thermal_scene() -> bool:
    """
    Copia scene.xml a scene_thermal.xml usando el servicio SceneService.
    """
    scene_service.prepare_thermal_scene(SCENE_DIR)
    return True

@scene_router.post("/load_scene")
async def load_scene(file: UploadFile = File(...)) -> FileResponse:    
    scene_path = scene_service.load_scene(file)
    image_path = render_service.render_basic_scene(scene_path)
    return StreamingResponse(open(image_path, "rb"), media_type="image/png", headers={"Content-Disposition": "attachment; filename=scene_basic.png"})

@scene_router.get("/render_scene_rgb")
async def load_scene_advanced() -> FileResponse:
    image_path = render_service.render_basic_scene(SCENE_DIR)
    return StreamingResponse(open(image_path, "rb"), media_type="image/png", headers={"Content-Disposition": "attachment; filename=scene_advanced.png"})

@scene_router.get("/has_loaded_scene")
async def has_loaded_scene() -> bool:
    return scene_service.has_loaded_scene(SCENE_DIR)

