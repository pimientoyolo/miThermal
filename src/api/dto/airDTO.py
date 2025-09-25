from pydantic import BaseModel
from typing import List

class AirDTO(BaseModel):
    temperature: float
    attenuation: List[float]
    wavelengths: List[float]