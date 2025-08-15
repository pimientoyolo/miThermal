"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""
from src.api.dto.suggestDTO import SuggestDTO

from fastapi import APIRouter

obj_router = APIRouter(
    prefix="/object",
    tags=["objects"]
)

@obj_router.get("/suggest_objects")
async def suggest_objects() -> list[SuggestDTO]:
    
    return [
        SuggestDTO(id="1", suggest="Objeto 3D sugerido"),
        SuggestDTO(id="2", suggest="Otro objeto 3D"),
        SuggestDTO(id="3", suggest="Tercer objeto 3D")
    ]