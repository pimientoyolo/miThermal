"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""
from src.api.dto.suggestDTO import SuggestDTO
from src.mitsuba_core.object_utils import ObjectUtils
from src.api.service.obj_service import ObjService
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import APIRouter, Query
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

@obj_router.get("/file/id")
async def get_obj(
    object_id: str = Query(..., description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')")
) -> FileResponse:
    """
    Obtiene un archivo de objeto 3D específico.
    
    Args:
        object_id: ID del objeto que puede incluir subdirectorios
        
    Returns:
        Archivo del objeto 3D
    """
    response = object_service.get_object_by_id(object_id)
    return response