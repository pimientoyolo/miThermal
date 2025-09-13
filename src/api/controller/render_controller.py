"""
Render Controller - Placeholder
Controller para renderizado de escenas
"""

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from src.api.service.render_service import RenderService

render_service = RenderService()

render_router = APIRouter(
    prefix="/render",
    tags=["rendering"]
)

@render_router.get("/rgb")
async def load_scene_advanced() -> FileResponse:
    image_path = render_service.render_basic_scene(SCENE_DIR)
    return StreamingResponse(open(image_path, "rb"), media_type="image/jpg", headers={"Content-Disposition": "attachment; filename=scene_advanced.jpg"})

@render_router.get("/depth")
async def render_scene_depth() -> FileResponse:
    image_path = render_service.render_depth_image(scenes_directory + "/scene_depth.xml")
    return StreamingResponse(open(image_path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=depth.npy"})

@render_router.get("/thermal")
async def render_scene_thermal() -> FileResponse:
    image_path = render_service.render_thermal_image(scenes_directory + "/scene_thermal.xml")
    return StreamingResponse(open(image_path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=thermal.npy"})