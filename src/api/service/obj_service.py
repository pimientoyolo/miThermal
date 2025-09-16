import logging
from src.mitsuba_core.object_utils import ObjectUtils
import os
from fastapi.responses import FileResponse
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import src.config as config
from src.api.dto.objectDTO import ObjectDTO, UpdateObjectDTO
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
        object_path = config.OUTPUT_STATIC_DIR / object_id

        if not os.path.exists(object_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {object_path}")
        
        return object_path
    
    def get_object_info_by_id(self, object_id: str) -> ObjectDTO:
        
        self.validate_object_id_exists(object_id)
        config_scene = config.get_config_scene_dict()

        wavelengths, emissivity = self.object_utils.read_object_emissivity_file(object_id)

        reflection = 1.0 - emissivity
        
        # Buscar el objeto en la configuración de la escena
        object_config = None
        for key in config_scene.keys():
            if object_id in key or key.endswith(object_id):
                object_config = config_scene[key]
                break

        wavelengths = wavelengths/1000
        
        # Construir el DTO
        return ObjectDTO(
            id=object_id,
            temperature=object_config.get("temperature"),
            emissivity=emissivity.tolist(),
            reflection=reflection.tolist(),
            wavelengths=wavelengths.tolist()
        )

    def update_object_info(self, object_data: UpdateObjectDTO) -> ObjectDTO:
        object_id = object_data.id
        self.validate_object_id_exists(object_id)
        
        config_scene = config.get_config_scene_dict()
        
        if object_id not in config_scene:
            raise HTTPException(status_code=404, detail=f"Configuración no encontrada para el objeto: {object_id}")
        
        # Actualizar la configuración del objeto
        config_scene[object_id]["temperature"] = object_data.temperature

        # Validar temperatura
        if object_data.temperature <= 0:
            raise HTTPException(status_code=400, detail="La temperatura debe ser superior a 0 (cero absoluto no permitido)")
        
        # Guardar la configuración actualizada
        config.save_config_scene_dict(config_scene)

        scene_service.update_thermal_scene_obj(object_id=object_id)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        
        return self.get_object_info_by_id(object_id)
    
    def update_object_emissivity(self, object_id: str, file: UploadFile) -> ObjectDTO:
        # Asegurar que el objeto existe en la escena (archivo 3D presente)
        self.validate_object_id_exists(object_id)

        # Validar archivo de entrada
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No se envió archivo")
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="El archivo debe ser .txt")

        raw = file.file.read()
        if not raw or len(raw) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")

        # Parsear como TAB-delimited: columnas [wavelength_um, reflectance_%]
        try:
            import io
            data = np.loadtxt(io.StringIO(raw.decode('utf-8')), delimiter='\t')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error leyendo el archivo (debe estar separado por TABs): {e}")

        # Normalizar a 2D si viene una sola fila
        if data.ndim == 1:
            if data.size < 2:
                raise HTTPException(status_code=400, detail="El archivo debe tener exactamente dos columnas: wavelength_um y reflectance_%")
            data = data.reshape(1, -1)

        # Debe tener exactamente dos columnas
        if data.shape[1] != 2:
            raise HTTPException(status_code=400, detail="El archivo debe tener exactamente dos columnas: wavelength_um y reflectance_%")

        wavelengths_um = data[:, 0].astype(float)
        reflectance_pct = data[:, 1].astype(float)

        # Validaciones de contenido
        if np.any(~np.isfinite(wavelengths_um)) or np.any(~np.isfinite(reflectance_pct)):
            raise HTTPException(status_code=400, detail="El archivo contiene valores no numéricos o infinitos")
        # Longitudes de onda (µm) deben ser > 0.0
        if np.any(wavelengths_um <= 0.0):
            raise HTTPException(status_code=400, detail="Las longitudes de onda (µm) deben ser mayores a 0")
        # Reflectancia en porcentaje 0..100
        if np.any(reflectance_pct < 0.0) or np.any(reflectance_pct > 100.0):
            raise HTTPException(status_code=400, detail="La reflectancia (%) debe estar entre 0 y 100")

        # Construir ruta de destino debajo de OUTPUT_STATIC_DIR conservando subcarpetas del object_id
        base_out = config.OUTPUT_STATIC_DIR
        rel_obj_id = object_id
        if os.path.isabs(object_id):
            rel_obj_id = os.path.splitdrive(object_id)[1].lstrip("\\/")
        dst_no_ext = os.path.splitext(rel_obj_id)[0]
        dst_path = os.path.join(base_out, dst_no_ext + ".txt")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Guardar el archivo validado "tal cual" (preservando tabs y formato)
        with open(dst_path, "wb") as f:
            f.write(raw)

        # Actualizar escenas afectadas
        scene_service.update_thermal_scene_obj(object_id=object_id)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()

        # Devolver DTO actualizado
        return self.get_object_info_by_id(object_id)

    def get_object_emissivity_file_by_id(self, object_id: str) -> FileResponse:
        self.validate_object_id_exists(object_id)

        # Construir ruta del archivo de emisividad debajo de OUTPUT_STATIC_DIR
        base_out = config.OUTPUT_STATIC_DIR
        rel_obj_id = object_id
        if os.path.isabs(object_id):
            rel_obj_id = os.path.splitdrive(object_id)[1].lstrip("\\/")
        dst_no_ext = os.path.splitext(rel_obj_id)[0]
        dst_path = os.path.join(base_out, dst_no_ext + ".txt")

        if not os.path.exists(dst_path):
            raise HTTPException(status_code=404, detail=f"No se encontró archivo de emisividad para el objeto: {object_id}")

        return FileResponse(dst_path, filename=os.path.basename(dst_path))