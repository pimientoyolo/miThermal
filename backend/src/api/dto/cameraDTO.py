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

class UpdateCameraDTO(BaseModel):
    spp : int
    width : int
    height : int
    rotate_x : float
    rotate_y : float
    rotate_z : float
    translate_x : float
    translate_y : float
    translate_z : float
    fov : float

class CameraInterpolationDTO(BaseModel):
    """DTO para solicitudes de interpolación de cámara"""
    origin: List[float]  # [x, y, z] posición inicial de la cámara
    end: List[float]     # [x, y, z] posición final de la cámara
    tracked_point: List[float]  # [x, y, z] punto objetivo que la cámara mira
    num_steps: int = 30  # número de frames a generar


class SphericalCameraInterpolationDTO(BaseModel):
    """DTO para interpolación esférica de cámara alrededor de un punto."""
    start_theta: float
    end_theta: float
    start_azimuth: float
    end_azimuth: float
    radius: float
    tracked_point: List[float]
    num_steps: int = 30
    lock_azimuth_to_end: bool = False
