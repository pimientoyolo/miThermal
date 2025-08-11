from fastapi import APIRouter
from src.api.service.scene_service import SceneService
from src.api.service.render_service import RenderService
from fastapi.responses import FileResponse

scene_service = SceneService()
render_service = RenderService()

scene_router = APIRouter(
    prefix="/scene",
    tags=["scene"]
)

@scene_router.post("/load_scene")
async def load_scene() -> FileResponse:
    scene_service.load_scene()
    image_path = render_service.render_basic_scene()
    return FileResponse(image_path, media_type="image/png", filename="scene_basic.png")
