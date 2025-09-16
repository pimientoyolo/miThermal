import logging
from src.mitsuba_core.object_utils import ObjectUtils
import os
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import src.config as config
from src.api.dto.objectDTO import ObjectDTO
from src.api.dto.airDTO import AirDTO
import numpy as np
from src.api.service.scene_service import SceneService

logger = logging.getLogger(__name__)

scene_service = SceneService()

class ObjService:
    def __init__(self):
        self.logger = logger
        self.object_utils = ObjectUtils()

    def get_object_file_by_id(self, object_id: str) -> FileResponse:

        object_path = self.validate_object_id_exists(object_id)

        if not os.path.exists(object_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {object_path}")

        file_extension = os.path.splitext(object_id)[1].lower()
        
        if file_extension == '.obj':
            return self.get_obj_response(object_path)
        
        elif file_extension == '.ply':
            path_obj = self.object_utils.ply2obj(object_path)
            return self.get_obj_response(path_obj)

        else:
            raise HTTPException(status_code=400, detail="Solo se soporta archivos .obj y .ply")

    def get_obj_response(self, path: str) -> StreamingResponse:
        def generate():
            with open(path, 'rb') as file:
                while chunk := file.read(8192):
                    yield chunk

        return StreamingResponse(
            generate(),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(path)}"}
        )
    
    def validate_object_id_exists(self, object_id: str) -> str:
        object_path = config.OUTPUT_STATIC_DIR + "/" + object_id

        if not os.path.exists(object_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {object_path}")
        
        return object_path
    
    def get_object_info_by_id(self, object_id: str) -> ObjectDTO:
        
        self.validate_object_id_exists(object_id)
        config_scene = config.get_config_scene_dict()
        
        # Buscar el objeto en la configuración de la escena
        object_config = None
        for key in config_scene.keys():
            if object_id in key or key.endswith(object_id):
                object_config = config_scene[key]
                break
        
        # Si no se encuentra en la config, usar valores por defecto
        if object_config is None:
            raise HTTPException(status_code=404, detail=f"Configuración no encontrada para el objeto: {object_id}")
        
        # Crear listas de emisividad y reflección
        emission = object_config.get("emissivity")
        emissivity_array = np.array(emission, dtype=float) if emission else np.array([])
        emissivity = emissivity_array.tolist()
        reflection_array = 1.0 - emissivity_array if len(emissivity_array) > 0 else np.array([])
        reflection = reflection_array.tolist()
        
        # Construir el DTO
        return ObjectDTO(
            id=object_id,
            temperature=object_config.get("temperature"),
            emissivity=emissivity,
            reflection=reflection
        )
    
    def update_object_info(self, object_data: ObjectDTO) -> ObjectDTO:
        object_id = object_data.id
        self.validate_object_id_exists(object_id)
        
        config_scene = config.get_config_scene_dict()
        
        if object_id not in config_scene:
            raise HTTPException(status_code=404, detail=f"Configuración no encontrada para el objeto: {object_id}")
        
        # Actualizar la configuración del objeto
        config_scene[object_id]["temperature"] = object_data.temperature
        config_scene[object_id]["emissivity"] = object_data.emissivity
        object_data.reflection = (1.0 - np.array(object_data.emissivity)).tolist()

        # Validar temperatura
        if object_data.temperature <= 0:
            raise HTTPException(status_code=400, detail="La temperatura debe ser superior a 0 (cero absoluto no permitido)")

        # Validar emisividad
        if not all(0 < emis <= 1 for emis in object_data.emissivity):
            raise HTTPException(status_code=400, detail="Todos los valores de emisividad deben estar entre 0 y 1")
        
        # Guardar la configuración actualizada
        config.save_config_scene_dict(config_scene)

        scene_service.update_thermal_scene_obj(object_id=object_id)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        
        return object_data
    
