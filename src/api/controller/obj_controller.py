"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""
from src.api.dto.suggestDTO import SuggestDTO
from src.mitsuba_core.object_utils import ObjectUtils
from src.api.service.obj_service import ObjService
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import APIRouter
from src.config import SCENE_DIR, get_output_path

object_utils = ObjectUtils()
object_service = ObjService()

obj_router = APIRouter(
    prefix="/object",
    tags=["objects"]
)

@obj_router.get("/suggest")
async def suggest_objects() -> list[SuggestDTO]:
    objects = object_utils.get_suggested_object(SCENE_DIR)
    return objects

@obj_router.get("/{object_id}")
async def get_obj(object_id: str) -> FileResponse:
    response = object_service.get_object(str(get_output_path("static") / object_id))
    return response