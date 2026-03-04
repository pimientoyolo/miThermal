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

class CameraFrameDTO(BaseModel):
    """DTO para un frame de cámara en la animación"""
    translate_x: float
    translate_y: float
    translate_z: float
    rotate_x: float
    rotate_y: float
    rotate_z: float

