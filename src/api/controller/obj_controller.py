"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""
from src.api.dto.objectDTO import ObjectDTO, UpdateObjectDTO
from src.api.dto.suggestDTO import SuggestDTO
from src.mitsuba_core.object_utils import ObjectUtils
from src.api.service.obj_service import ObjService
from fastapi.responses import FileResponse
from fastapi import APIRouter, File, Query, UploadFile, Form
from src.config import SCENE_DIR
import json


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
async def get_obj_file(
    object_id: str = Query(..., description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')")
) -> FileResponse:
    """
    Obtiene un archivo de objeto 3D específico.
    
    Args:
        object_id: ID del objeto que puede incluir subdirectorios
        
    Returns:
        Archivo del objeto 3D
    """
    response = object_service.get_object_file_by_id(object_id)
    return response

@obj_router.get("/id")
async def get_obj_info(
    object_id: str = Query(..., description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')")
) -> ObjectDTO:
    """
    Obtiene la información de un objeto 3D específico.
    
    Args:
        object_id: ID del objeto que puede incluir subdirectorios
        
    Returns:
        Información del objeto 3D
    """
    response = object_service.get_object_info_by_id(object_id)
    return response

@obj_router.put("/id")
async def update_obj_info(
    object_data: UpdateObjectDTO
) -> ObjectDTO:
    """
    Actualiza la información de un objeto 3D específico.
    
    Args:
        object_data: Datos del objeto a actualizar
        
    Returns:
        Información actualizada del objeto 3D
    """
    object_data = object_service.update_object_info(object_data)
    
    return object_data

@obj_router.put("/update-with-emissivity")
async def update_with_emissivity(
    object_data_json: str = Form(..., description="JSON con la lista de objetos y temperaturas"),
    emissivity_file: UploadFile = File(..., description="Archivo de emisividad")
) -> str:
    """
    Actualiza la temperatura de varios objetos 3D y sube un archivo de emisividad.
    """
    try:
        object_data_list = [UpdateObjectDTO(**obj) for obj in json.loads(object_data_json)]
    except Exception as e:
        return f"Error al parsear el JSON: {e}"

    result = object_service.update_objects_with_emissivity(object_data_list, emissivity_file)
    return result


@obj_router.get("/emissivity/id")
async def get_obj_emissivity_file(
    object_id: str = Query(..., description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')")
) -> FileResponse:
    """
    Obtiene el archivo de emisividad de un objeto 3D específico.

    Args:
        object_id: ID del objeto que puede incluir subdirectorios

    Returns:
        Archivo de emisividad del objeto 3D
    """
    response = object_service.get_object_emissivity_file_by_id(object_id)
    return response

@obj_router.put("/emissivity/id")
async def update_obj_emissivity(
    object_id: str = Query(..., description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')"),
    file: UploadFile = File(...)
) -> str:
    """
    Actualiza la emisividad de un objeto 3D específico.

    Args:
        object_id: ID del objeto que puede incluir subdirectorios
        emissivity: Lista de valores de emisividad a actualizar

    Returns:
        Mensaje de confirmación y advertencias si las hay
    """
    mensaje = object_service.update_object_emissivity(object_id, file)

    return mensaje

@obj_router.get("/suggest/emissivity")
async def suggest_object_emissivity() -> list[str]:
    objects = object_service.get_suggested_object_emissivity()
    return objects

@obj_router.put("/default/emissivity")
async def update_default_emissivity(
    file_name: str = Query(..., description="Nombre del archivo de emisividad (ej: 'default.txt')"),
    object_id: str = Query(..., description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')"),
) -> FileResponse:
    """
    Actualiza la emisividad por defecto.

    Args:
        file: Archivo que contiene la nueva emisividad por defecto

    Returns:
        Mensaje de confirmación
    """
    object_service.update_default_emissivity(file_name, object_id)
    return object_service.get_object_emissivity_file_by_id(object_id)
