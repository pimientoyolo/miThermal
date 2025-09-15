from pydantic import BaseModel
from typing import List

class ObjectDTO(BaseModel):
    id: str
    temperature: float
    emissivity: List[float]
    reflection: List[float]