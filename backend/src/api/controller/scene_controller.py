import logging
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, Body, HTTPException
from fastapi import Query
from src.api.service.scene_service import SceneService
from src.api.service.render_service import RenderService
from fastapi.responses import FileResponse
from src.config import PathManager
from src.api.dto.cameraDTO import (
    CameraInterpolationDTO,
    SphericalCameraInterpolationDTO,
    CameraAnimationConfigDTO,
)
from typing import List


logger = logging.getLogger(__name__)

scene_service = SceneService()
render_service = RenderService()
path_manager = PathManager

scene_router = APIRouter(
    prefix="/scene",
    tags=["scene"]
)

@scene_router.post("/load")
async def load_scene(file: UploadFile = File(...)) -> FileResponse:    
    request_id = uuid4().hex
    logger.info(
        "POST /scene/load iniciado request_id=%s filename=%s",
        request_id,
        file.filename,
    )

    try:
        scene_service.load_scene(file)
        render_service.render_basic_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_thermal_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()
        result_path = path_manager.get_result_path("rgb")
        logger.info("POST /scene/load completado request_id=%s", request_id)
        return FileResponse(
            result_path,
            media_type="image/png",
            filename="rgb.png",
            headers={"X-Request-ID": request_id},
        )
    except HTTPException as exc:
        logger.exception(
            "POST /scene/load error controlado request_id=%s status=%s",
            request_id,
            exc.status_code,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"[request_id={request_id}] {exc.detail}",
            headers=exc.headers,
        ) from exc
    except Exception as exc:
        logger.exception("POST /scene/load error inesperado request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"[request_id={request_id}] Error al cargar escena: {exc}",
            headers={"X-Request-ID": request_id},
        ) from exc


@scene_router.get("/loaded")
async def has_loaded_scene() -> bool:
    scene_path = path_manager.get_scene_path("rgb")
    return scene_service.has_loaded_scene(scene_path)

@scene_router.get("/miThermal")
async def get_full_scene() -> FileResponse:
    return scene_service.get_scene_mi_thermal()

@scene_router.post("/miThermal")
async def create_mi_thermal_scene(
    file: UploadFile = File(...)
) -> FileResponse:
    scene_service.upload_mi_thermal_scene(file)
    render_service.render_basic_scene()
    result_path = path_manager.get_result_path("rgb")
    return FileResponse(result_path, media_type="image/png", filename="rgb.png")

@scene_router.get("/suggest")
async def suggest_mi_thermal_scene() -> list[str]:  
    return scene_service.get_suggested_mi_thermal_scene()

@scene_router.post("/select/default")
async def set_default_scene(
    file_name: str = Query(..., description="archivo de la escena sugerida (ej: 'room.zip')") # type: ignore
) -> FileResponse:
    scene_service.set_default_scene(file_name)
    render_service.render_basic_scene()
    result_path = path_manager.get_result_path("rgb")
    return FileResponse(result_path, media_type="image/png", filename="rgb.png")

@scene_router.post("/camera/interpolation")
async def generate_camera_interpolation(
    data: CameraInterpolationDTO = Body(...)
) -> List[dict]:
    """
    Genera una interpolación de cámara para animación.
    
    Args:
        data: Datos de interpolación (origin, end, tracked_point, num_steps)
        
    Returns:
        Lista de configuraciones de cámara para cada frame
    """
    return scene_service.generate_camera_animation(
        origin=data.origin,
        end=data.end,
        tracked_point=data.tracked_point,
        num_steps=data.num_steps
    )

@scene_router.post("/camera/animation/render")
async def render_camera_animation(
    data: CameraInterpolationDTO = Body(...)
) -> FileResponse:
    """
    Genera y renderiza una animación completa de cámara.
    
    Genera los frames de interpolación y renderiza cada uno (RGB, depth, thermal, etc.).
    Los resultados se guardan en output/renders/animation/ y se devuelven en un ZIP.
    
    Args:
        data: Datos de interpolación (origin, end, tracked_point, num_steps)
        
    Returns:
        Archivo ZIP con todos los frames renderizados
    """
    # Generar frames de cámara
    camera_frames = scene_service.generate_camera_animation(
        origin=data.origin,
        end=data.end,
        tracked_point=data.tracked_point,
        num_steps=data.num_steps
    )
    
    # Renderizar secuencia completa
    zip_path = scene_service.render_camera_animation_sequence(
        camera_frames,
        spp=data.spp,
        width=data.width,
        height=data.height,
        num_bands=data.num_bands
    )
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="camera_animation.zip"
    )


@scene_router.post("/camera/animation/preview")
async def render_camera_animation_preview(
    data: CameraInterpolationDTO = Body(...)
) -> FileResponse:
    """
    Genera un preview rápido en GIF de la trayectoria de cámara.

    - Mitad de frames
    - Baja resolución y SPP
    - Render RGB únicamente
    """
    camera_frames = scene_service.generate_camera_animation(
        origin=data.origin,
        end=data.end,
        tracked_point=data.tracked_point,
        num_steps=data.num_steps
    )

    gif_path = scene_service.render_camera_path_preview_gif(camera_frames)

    return FileResponse(
        gif_path,
        media_type="image/gif",
        filename="camera_path_preview.gif"
    )


@scene_router.post("/camera/interpolation/spherical")
async def generate_camera_interpolation_spherical(
    data: SphericalCameraInterpolationDTO = Body(...)
) -> List[dict]:
    """Genera interpolación esférica de cámara alrededor de un punto objetivo."""
    return scene_service.generate_camera_animation_spherical(
        start_theta=data.start_theta,
        end_theta=data.end_theta,
        start_azimuth=data.start_azimuth,
        end_azimuth=data.end_azimuth,
        radius=data.radius,
        start_radius=data.start_radius,
        end_radius=data.end_radius,
        tracked_point=data.tracked_point,
        num_steps=data.num_steps,
        lock_azimuth_to_end=data.lock_azimuth_to_end,
        theta_expr=data.theta_expr,
        azimuth_expr=data.azimuth_expr,
        radius_expr=data.radius_expr,
    )


@scene_router.post("/camera/animation/render/spherical")
async def render_camera_animation_render_spherical(
    data: SphericalCameraInterpolationDTO = Body(...)
) -> FileResponse:
    """Genera y renderiza una animación completa usando interpolación esférica."""
    camera_frames = scene_service.generate_camera_animation_spherical(
        start_theta=data.start_theta,
        end_theta=data.end_theta,
        start_azimuth=data.start_azimuth,
        end_azimuth=data.end_azimuth,
        radius=data.radius,
        start_radius=data.start_radius,
        end_radius=data.end_radius,
        tracked_point=data.tracked_point,
        num_steps=data.num_steps,
        lock_azimuth_to_end=data.lock_azimuth_to_end,
        theta_expr=data.theta_expr,
        azimuth_expr=data.azimuth_expr,
        radius_expr=data.radius_expr,
    )
    zip_path = scene_service.render_camera_animation_sequence(
        camera_frames,
        spp=data.spp,
        width=data.width,
        height=data.height,
        num_bands=data.num_bands
    )
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="camera_animation_spherical.zip",
    )


@scene_router.post("/camera/animation/preview/spherical")
async def render_camera_animation_preview_spherical(
    data: SphericalCameraInterpolationDTO = Body(...)
) -> FileResponse:
    """Genera un preview GIF usando interpolación esférica."""
    camera_frames = scene_service.generate_camera_animation_spherical(
        start_theta=data.start_theta,
        end_theta=data.end_theta,
        start_azimuth=data.start_azimuth,
        end_azimuth=data.end_azimuth,
        radius=data.radius,
        start_radius=data.start_radius,
        end_radius=data.end_radius,
        tracked_point=data.tracked_point,
        num_steps=data.num_steps,
        lock_azimuth_to_end=data.lock_azimuth_to_end,
        theta_expr=data.theta_expr,
        azimuth_expr=data.azimuth_expr,
        radius_expr=data.radius_expr,
    )
    gif_path = scene_service.render_camera_path_preview_gif(camera_frames)
    return FileResponse(
        gif_path,
        media_type="image/gif",
        filename="camera_path_preview_spherical.gif",
    )