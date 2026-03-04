"""
Render Controller - Placeholder
Controller para renderizado de escenas
"""

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from src.api.service.render_service import RenderService
from src.config import PathManager
import aiofiles

OCTET_STREAM_MEDIA_TYPE = "application/octet-stream"

render_service = RenderService()
path_manager = PathManager

render_router = APIRouter(
    prefix="/render",
    tags=["rendering"]
)

@render_router.get("/rgb")
async def load_scene_advanced() -> FileResponse:
    render_service.render_basic_scene()
    result_path = path_manager.get_result_path("rgb")
    return FileResponse(result_path, media_type="image/png", filename="rgb.png")

@render_router.get("/depth")
async def render_scene_depth() -> StreamingResponse:
    render_service.render_depth_image()
    result_path = path_manager.get_result_path("depth")
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=depth.npy"})

@render_router.get("/thermal")
async def render_scene_thermal() -> StreamingResponse:
    render_service.render_thermal_image()
    result_path = path_manager.get_result_path("thermal")
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=thermal.npy"})

@render_router.get("/air/blackbody")
async def render_scene_blackbody_air() -> StreamingResponse:
    render_service.render_blackbody_air_image()
    result_path = path_manager.get_result_path("blackbody_air")
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=blackbody_air.npy"})

@render_router.get("/air/transmittance")
async def render_scene_transmittance_blackbody_air() -> StreamingResponse:
    render_service.render_transmittance_blackbody_air_image()
    result_path = path_manager.get_result_path("transmittance_blackbody_air")
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=transmittance_blackbody_air.npy"})

@render_router.get("/air/contribution")
async def render_scene_contribution_blackbody_air() -> StreamingResponse:
    render_service.render_contribution_air()
    result_path = path_manager.get_result_path("contribution_blackbody_air")
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=contribution_blackbody_air.npy"})

@render_router.get("/temperature/map")
async def render_scene_temperature_map() -> StreamingResponse:
    render_service.render_temperature_map()
    result_path = path_manager.get_result_path("temperature_map")
    async def iterfile():
        async with aiofiles.open(result_path, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=temperature_map.npy"})