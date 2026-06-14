from pydantic import BaseModel
from typing import List

class ObjectDTO(BaseModel):
    id: str
    temperature: float
    emissivity: List[float]
    reflection: List[float]
    wavelengths: List[float]
    material_type: str = "diffuse"
    roughness: float = 0.05

class UpdateObjectDTO(BaseModel):
    id: str
    temperature: float