from pydantic import BaseModel
from typing import List, Optional

class CameraDTO(BaseModel):
    spp : int
    width : int
    height : int