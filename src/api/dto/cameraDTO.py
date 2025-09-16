from pydantic import BaseModel
from typing import List, Optional

class CameraDTO(BaseModel):
    spp : int
    width : int
    height : int
    wavelengths : List[float]
    num_bands : int

class UpdateCameraDTO(BaseModel):
    spp : int
    width : int
    height : int