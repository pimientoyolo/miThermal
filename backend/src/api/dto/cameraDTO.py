from pydantic import BaseModel
from typing import List

class CameraDTO(BaseModel):
    spp : int
    width : int
    height : int
    wavelengths : List[float]
    num_bands : int
    rotate_x : float
    rotate_y : float
    rotate_z : float
    translate_x : float
    translate_y : float
    translate_z : float
    fov : float
    # Parametrización esférica (opcional, calculada a partir de cartesianas si no se provee)
    theta: float | None = None
    phi: float | None = None
    radius: float | None = None
    target_x: float | None = 0.0
    target_y: float | None = 0.0
    target_z: float | None = 0.0
    # Vector UP explícito para evitar singularidades (opcional)
    up_x: float | None = None
    up_y: float | None = None
    up_z: float | None = None

class UpdateCameraDTO(BaseModel):
    spp : int
    width : int
    height : int
    rotate_x : float | None = None
    rotate_y : float | None = None
    rotate_z : float | None = None
    translate_x : float | None = None
    translate_y : float | None = None
    translate_z : float | None = None
    fov : float
    # Nuevos parámetros esféricos
    theta: float | None = None
    phi: float | None = None
    radius: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    target_z: float | None = None
    up_x: float | None = None
    up_y: float | None = None
    up_z: float | None = None

class CameraSpatialConfigDTO(BaseModel):
    """DTO para exportar/importar solo la configuración espacial de la cámara"""
    rotate_x: float
    rotate_y: float
    rotate_z: float
    translate_x: float
    translate_y: float
    translate_z: float
    fov: float
    theta: float | None = None
    phi: float | None = None
    radius: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    target_z: float | None = None
    up_x: float | None = None
    up_y: float | None = None
    up_z: float | None = None

class CameraInterpolationDTO(BaseModel):
    """DTO para solicitudes de interpolación de cámara"""
    origin: List[float]  # [x, y, z] posición inicial de la cámara
    end: List[float]     # [x, y, z] posición final de la cámara
    tracked_point: List[float]  # [x, y, z] punto objetivo que la cámara mira
    num_steps: int = 30  # número de frames a generar
    # Parámetros de renderizado opcionales para la animación
    spp: int | None = None
    width: int | None = None
    height: int | None = None
    num_bands: int | None = None


class SphericalCameraInterpolationDTO(BaseModel):
    """DTO para interpolación esférica de cámara alrededor de un punto."""
    start_theta: float
    end_theta: float
    start_azimuth: float
    end_azimuth: float
    start_radius: float | None = None
    end_radius: float | None = None
    radius: float | None = None # Mantener para retrocompatibilidad
    tracked_point: List[float]
    num_steps: int = 30
    lock_azimuth_to_end: bool = False
    # Expresiones personalizadas para la trayectoria (funciones de 't' de 0 a 1)
    theta_expr: str | None = None
    azimuth_expr: str | None = None
    radius_expr: str | None = None
    # Auto-FOV basado en radio
    auto_fov: bool = False
    initial_fov: float | None = None
    # Parámetros de renderizado opcionales para la animación
    spp: int | None = None
    width: int | None = None
    height: int | None = None
    num_bands: int | None = None

class CameraAnimationConfigDTO(BaseModel):
    """DTO para exportar/importar la configuración de una animación de cámara completa"""
    mode: str # "linear" o "spherical"
    linear_data: CameraInterpolationDTO | None = None
    spherical_data: SphericalCameraInterpolationDTO | None = None
