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
