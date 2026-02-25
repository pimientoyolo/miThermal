from pydantic import BaseModel
from typing import List

class ObjectDTO(BaseModel):
    id: str
    temperature: float
    emissivity: List[float]
    reflection: List[float]
    wavelengths: List[float]

class UpdateObjectDTO(BaseModel):
    id: str
    temperature: float