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

@scene_router.post("/load")
async def load_scene(file: UploadFile = File(...)) -> FileResponse:    
    scene_path = scene_service.load_scene(file)
    image_path = render_service.render_basic_scene(scene_path)
    return StreamingResponse(open(image_path, "rb"), media_type="image/jpg", headers={"Content-Disposition": "attachment; filename=scene_basic.jpg"})


@scene_router.get("/loaded")
async def has_loaded_scene() -> bool:
    return scene_service.has_loaded_scene(SCENE_DIR)

