"""
Render Controller - Placeholder
Controller para renderizado de escenas
"""

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from src.api.service.render_service import RenderService
import src.config as config
import aiofiles

OCTET_STREAM_MEDIA_TYPE = "application/octet-stream"

render_service = RenderService()

render_router = APIRouter(
    prefix="/render",
    tags=["rendering"]
)

@render_router.get("/rgb")
async def load_scene_advanced() -> FileResponse:
    render_service.render_basic_scene()
    return FileResponse(config.IMAGE_DIR, media_type="image/png", filename="rgb.png")

@render_router.get("/depth")
async def render_scene_depth() -> StreamingResponse:
    render_service.render_depth_image()
    async def iterfile():
        async with aiofiles.open(config.DEPTH_DIR, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=depth.npy"})

@render_router.get("/thermal")
async def render_scene_thermal() -> StreamingResponse:
    render_service.render_thermal_image()
    async def iterfile():
        async with aiofiles.open(config.THERMAL_DIR, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=thermal.npy"})

@render_router.get("/air/blackbody")
async def render_scene_blackbody_air() -> StreamingResponse:
    render_service.render_blackbody_air_image()
    async def iterfile():
        async with aiofiles.open(config.BLACKBODY_AIR_DIR, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=blackbody_air.npy"})

@render_router.get("/air/transmittance")
async def render_scene_transmittance_blackbody_air() -> StreamingResponse:
    render_service.render_transmittance_blackbody_air_image()
    async def iterfile():
        async with aiofiles.open(config.TRANSMITTANCE_BLACKBODY_AIR_DIR, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=transmittance_blackbody_air.npy"})

@render_router.get("/air/contribution")
async def render_scene_contribution_blackbody_air() -> StreamingResponse:
    render_service.render_contribution_air()
    async def iterfile():
        async with aiofiles.open(config.CONTRIBUTION_BLACKBODY_AIR_DIR, "rb") as file:
            while chunk := await file.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type=OCTET_STREAM_MEDIA_TYPE, headers={"Content-Disposition": "attachment; filename=contribution_blackbody_air.npy"})