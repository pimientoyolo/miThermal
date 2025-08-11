from fastapi import APIRouter
from src.api.service.scene_service import SceneService

scene_service = SceneService()

scene_router = APIRouter(
    prefix="/scene",
    tags=["scene"]
)

@scene_router.post("/load_scene")
async def load_scene():
    return {"message": "Escena cargada", "data": "data cargada"}
