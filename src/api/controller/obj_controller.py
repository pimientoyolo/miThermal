"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""
from src.api.dto.suggestDTO import SuggestDTO
from src.mitsuba_core.object_utils import ObjectUtils
from fastapi import APIRouter
from src.config import SCENE_DIR
object_utils = ObjectUtils()

obj_router = APIRouter(
    prefix="/object",
    tags=["objects"]
)

@obj_router.get("/suggest_objects")
async def suggest_objects() -> list[SuggestDTO]:
    objects = object_utils.get_suggested_object(SCENE_DIR)
    return objects
