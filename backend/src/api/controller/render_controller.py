"""
Render Controller - Placeholder
Controller para renderizado de escenas
"""

from typing import Callable
from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from src.api.service.render_service import RenderService
from src.api.service.scene_service import SceneService
from src.config import PathManager
import aiofiles

OCTET_STREAM_MEDIA_TYPE = "application/octet-stream"

render_service = RenderService()
scene_service = SceneService()
path_manager = PathManager

render_router = APIRouter(
    prefix="/render",
    tags=["rendering"]
)


async def _render_and_stream_npy(
    render_func: Callable[[], None],
    result_type: str,
    filename: str
) -> StreamingResponse:
    """Helper para renderizar y hacer streaming de archivos .npy"""
    scene_service.update_scene_camera_all()
    render_func()
    result_path = path_manager.get_result_path(result_type)
    
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type=OCTET_STREAM_MEDIA_TYPE,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@render_router.get("/rgb")
async def load_scene_advanced() -> FileResponse:
    scene_service.update_scene_camera_all()
    render_service.render_basic_scene()
    result_path = path_manager.get_result_path("rgb")
    return FileResponse(result_path, media_type="image/png", filename="rgb.png")

@render_router.get("/depth")
async def render_scene_depth() -> StreamingResponse:
    return await _render_and_stream_npy(
        render_service.render_depth_image,
        "depth",
        "depth.npy"
    )

@render_router.get("/thermal")
async def render_scene_thermal() -> StreamingResponse:
    return await _render_and_stream_npy(
        render_service.render_thermal_image,
        "thermal",
        "thermal.npy"
    )

@render_router.get("/thermal/raw")
async def render_scene_thermal_raw() -> StreamingResponse:
    """Retorna la radiancia de superficie (L_surface) sin contribución de aire"""
    return await _render_and_stream_npy(
        render_service.render_thermal_image,
        "thermal_raw",
        "thermal_raw.npy"
    )

@render_router.get("/air/blackbody")
async def render_scene_blackbody_air() -> StreamingResponse:
    return await _render_and_stream_npy(
        render_service.render_blackbody_air_image,
        "blackbody_air",
        "blackbody_air.npy"
    )

@render_router.get("/air/transmittance")
async def render_scene_transmittance_blackbody_air() -> StreamingResponse:
    return await _render_and_stream_npy(
        render_service.render_transmittance_blackbody_air_image,
        "transmittance_blackbody_air",
        "transmittance_blackbody_air.npy"
    )

@render_router.get("/air/contribution")
async def render_scene_contribution_blackbody_air() -> StreamingResponse:
    return await _render_and_stream_npy(
        render_service.render_contribution_air,
        "contribution_blackbody_air",
        "contribution_blackbody_air.npy"
    )

@render_router.get("/temperature/map")
async def render_scene_temperature_map() -> StreamingResponse:
    return await _render_and_stream_npy(
        render_service.render_temperature_map,
        "temperature_map",
        "temperature_map.npy"
    )