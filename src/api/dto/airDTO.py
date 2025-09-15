from pydantic import BaseModel
from typing import List, Optional

class AirDTO(BaseModel):
    temperature: float
    attenuation: List[float]
    wavelengths: List[float]
    num_bands: int