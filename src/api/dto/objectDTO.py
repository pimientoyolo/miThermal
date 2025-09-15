from pydantic import BaseModel
from typing import List, Optional

class ObjectDTO(BaseModel):
    id: str
    temperature: float
    emissivity: List[float]
    reflection: Optional[List[float]] = None