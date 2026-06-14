"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""
from src.api.dto.objectDTO import ObjectDTO, UpdateObjectDTO
from src.api.dto.suggestDTO import SuggestDTO
from src.utils.objects.objects import ObjectUtils
from src.api.service.obj_service import ObjService
from fastapi.responses import FileResponse
from fastapi import APIRouter, File, Query, UploadFile, Form, HTTPException
from src.config import PathManager
import json


object_utils = ObjectUtils()
object_service = ObjService()
path_manager = PathManager

obj_router = APIRouter(
    prefix="/object",
    tags=["objects"]
)

@obj_router.get("/suggest")
async def suggest_objects() -> list[SuggestDTO]:
    scene_rgb_path = path_manager.get_scene_path("rgb")
    objects = object_utils.get_suggested_object(scene_rgb_path)
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
    file_name: str = Query(
        ...,
        description="Nombre del archivo de emisividad (ej: 'default.txt')"
    ),
    object_id: str = Query(
        ...,
        description="ID del objeto (ej: 'meshes/objeto.ply', 'Dragon.obj')"
    ),
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


@obj_router.get("/families")
async def list_families():
    """
    Lista todas las familias de objetos detectadas.
    
    Detecta automáticamente familias usando patrones de nomenclatura:
    - Blender: object.001, object.002
    - General: object_001, object_002
    
    Returns:
        {
            "families": [
                {
                    "base_name": "tree_leaf",
                    "members": ["tree_leaf.001", "tree_leaf.002", ...],
                    "count": 50,
                    "shared_properties": {"temperature": 300, ...}
                },
                ...
            ],
            "statistics": {
                "total_families": 15,
                "total_objects_in_families": 200,
                "avg_family_size": 13.3,
                "largest_family": {"base_name": "tree_leaf", "count": 50}
            },
            "total_families": 15
        }
    """
    return object_service.get_families()


@obj_router.get("/families/preview/{object_id:path}")
async def preview_family_update(object_id: str):
    """
    Previsualiza qué objetos se actualizarían en modo familia.
    
    Args:
        object_id: ID del objeto para el que se quiere ver la familia
        
    Returns:
        Información de la familia y cuántos objetos se actualizarían
    """
    return object_service.preview_family_update(object_id)


@obj_router.put("/update-with-mode")
async def update_object_with_mode(
    object_id: str | None = Query(
        None,
        description="ID del objeto (query param, recomendado para IDs con '/')",
    ),
    mode: str = Query(
        ...,
        description="Modo de actualización: 'Objeto' o 'Familia'"
    ),
    temperature: float = Query(
        None,
        description="Nueva temperatura en Kelvin"
    ),
    emissivity_file: UploadFile | None = File(
        None,
        description="Archivo de emisividad"
    ),
    is_reflectance: bool = Query(
        False,
        description="Indica si el archivo es de reflectancia (se convertirá a emisividad: 1-R)"
    ),
    temp_min: float | None = Query(
        None,
        description="Temperatura mínima para rango aleatorio"
    ),
    temp_max: float | None = Query(
        None,
        description="Temperatura máxima para rango aleatorio"
    ),
    material_type: str | None = Query(
        None,
        description="Tipo de material: 'diffuse' o 'reflectante'"
    ),
    roughness: float | None = Query(
        None,
        description="Rugosidad (0.0 a 1.0) para materiales reflectantes"
    )
):
    """
    Actualiza objeto o familia según modo seleccionado.
    
    Args:
        object_id: ID del objeto seleccionado
        mode: "Objeto" para actualizar solo este, "Familia" para toda la familia
        temperature: Nueva temperatura (opcional)
        emissivity_file: Nueva emisividad (opcional)
        is_reflectance: Si el archivo es de reflectancia
        temp_min: Temperatura mínima del rango aleatorio (opcional)
        temp_max: Temperatura máxima del rango aleatorio (opcional)
        material_type: Tipo de material (opcional)
        roughness: Rugosidad (opcional)
        
    Returns:
        {
            "objects_updated": ["obj1", "obj2", ...],
            "count": 5,
            "mode": "family",
            "family_name": "tree_leaf",
            "properties_updated": ["temperature", "emissivity_file"]
        }
    """
    if not object_id:
        raise HTTPException(
            status_code=400,
            detail="Debe proporcionar object_id en query",
        )

    return await object_service.update_object_with_mode(
        object_id=object_id,
        mode=mode,
        temperature=temperature,
        emissivity_file=emissivity_file,
        is_reflectance=is_reflectance,
        temp_min=temp_min,
        temp_max=temp_max,
        material_type=material_type,
        roughness=roughness
    )
