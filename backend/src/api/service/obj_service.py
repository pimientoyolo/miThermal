import logging
from src.utils.objects.objects import ObjectUtils
from src.utils.objects.family_manager import FamilyManager
import os
from fastapi.responses import FileResponse
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.utils.decorators import handle_file_errors, log_execution
from src.config import (
    PathManager, 
    OUTPUT_STATIC_DIR, 
    OUTPUT_ASSETS_DIR,
    get_config_scene_dict, 
    save_config_scene_dict
)
from src.api.dto.objectDTO import ObjectDTO, UpdateObjectDTO
import numpy as np
from src.api.service.scene_service import SceneService
import shutil
import io
import hashlib
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
path_manager = PathManager

scene_service = SceneService()

class ObjService:
    def __init__(self):
        self.logger = logger
        self.object_utils = ObjectUtils()

    def get_object_file_by_id(self, object_id: str) -> FileResponse:

        object_path = self.validate_object_id_exists(object_id)

        logger.info(f"Archivo solicitado: {object_path}")

        if not os.path.exists(object_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {object_path}")

        file_extension = os.path.splitext(object_id)[1].lower()
        
        if file_extension == '.obj':
            return self.get_obj_response(object_path)
        
        elif file_extension == '.ply':
            try:
                path_obj = self.object_utils.ply2obj(object_path)
                return self.get_obj_response(path_obj)
            except HTTPException as exc:
                logger.warning(
                    "No se pudo convertir PLY a OBJ para '%s': %s. "/
                    "Se enviará el PLY original.",
                    object_id,
                    exc.detail,
                )
                return self.get_obj_response(object_path)

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
        object_path = OUTPUT_STATIC_DIR / object_id

        logger.debug(f"Validando existencia del objeto: {object_id} en ruta: {object_path}")

        if not os.path.exists(object_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {object_path}")
        
        return object_path
    
    def get_object_info_by_id(self, object_id: str) -> ObjectDTO:
        
        self.validate_object_id_exists(object_id)
        config_scene = get_config_scene_dict()

        wavelengths, emissivity = self.object_utils.read_object_emissivity_file(object_id)

        reflection = 1.0 - emissivity
        
        object_config = config_scene.get("objects", {}).get(object_id)

        # Si la configuración del objeto no existe, crear entrada por defecto
        if object_config is None:
            default_temp = 300
            # Intentar crear archivo de emisividad por defecto bajo OUTPUT_STATIC_DIR
            try:
                self.object_utils.save_default_emissivity(object_id)
            except Exception:
                # Si falla la copia, igual proseguimos con el archivo por defecto global
                pass

            # Construir ruta del archivo de emisividad relativo al OUTPUT_STATIC_DIR
            base_out = Path(OUTPUT_STATIC_DIR)
            rel_obj = Path(object_id)
            if rel_obj.is_absolute():
                rel_obj = rel_obj.relative_to(rel_obj.anchor)
            emissivity_path = str((base_out / rel_obj.with_suffix('.txt')))

            config_scene.setdefault("objects", {})[object_id] = {
                "temperature": default_temp,
                "emissivity_file": emissivity_path
            }
            save_config_scene_dict(config_scene)
            object_config = config_scene["objects"][object_id]

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
        
        config_scene = get_config_scene_dict()
        
        if object_id not in config_scene.get("objects", {}):
            raise HTTPException(status_code=404, detail=f"Configuración no encontrada para el objeto: {object_id}")
        
        # Actualizar la configuración del objeto
        config_scene["objects"][object_id]["temperature"] = object_data.temperature

        # Validar temperatura
        if object_data.temperature <= 0:
            raise HTTPException(status_code=400, detail="La temperatura debe ser superior a 0 (cero absoluto no permitido)")
        
        # Guardar la configuración actualizada
        save_config_scene_dict(config_scene)

        scene_service.update_thermal_scene_obj(object_id=object_id)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()
        
        return self.get_object_info_by_id(object_id)

    def update_object_emissivity(self, object_id: str, file: UploadFile) -> str:
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
        base_out = OUTPUT_STATIC_DIR
        rel_obj_id = object_id
        if os.path.isabs(object_id):
            rel_obj_id = os.path.splitdrive(object_id)[1].lstrip("\\/")
        dst_no_ext = os.path.splitext(rel_obj_id)[0]
        dst_path = os.path.join(base_out, dst_no_ext + ".txt")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Guardar el archivo validado "tal cual" (preservando tabs y formato)
        with open(dst_path, "wb") as f:
            f.write(raw)

        mensaje = ""

        scene_config = get_config_scene_dict()
        wavelengths_scene = scene_config.get("wavelengths")
        wavelengths_scene = np.array(wavelengths_scene)/1000  # Convertir a µm

        w_min = np.min(wavelengths_scene)
        w_max = np.max(wavelengths_scene)

        if np.min(wavelengths_um) > w_min and np.max(wavelengths_um) < w_max:
                mensaje =  "Advertencia: El archivo contiene longitudes de onda que no cubre el rango completo de la cámara. Se recomienda incluir valores entre {:.3f} µm y {:.3f} µm".format(w_min, w_max)
            
        elif np.min(wavelengths_um) > w_min:
            mensaje =  "Advertencia: El archivo contiene longitudes de onda mayores al mínimo de la cámara ({:.3f} µm). Se recomienda incluir valores menores".format(w_min)

        elif np.max(wavelengths_um) < w_max:
            mensaje =  "Advertencia: El archivo contiene longitudes de onda menores al máximo de la cámara ({:.3f} µm). Se recomienda incluir valores mayores".format(w_max)
        
        else:
            mensaje = "Archivo de atenuación del aire actualizado correctamente" 

        # Actualizar escenas afectadas
        scene_service.update_thermal_scene_obj(object_id=object_id)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()

        # Devolver mensaje
        return mensaje
    
    # from fastapi import UploadFile, HTTPException

    def update_objects_with_emissivity(self, object_data_list: list[UpdateObjectDTO], file: UploadFile) -> str:
        """
        Actualiza múltiples objetos 3D con sus temperaturas y un archivo de emisividad/atenuación único.
        """
        # 1. Validar archivo de emisividad
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No se envió archivo de emisividad")
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="El archivo debe ser .txt")

        raw = file.file.read()
        if not raw or len(raw) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")

        try:
            data = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter="\t")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error leyendo el archivo (TAB-delimited esperado): {e}")

        # Normalizar a 2D si viene una sola fila
        if data.ndim == 1:
            if data.size < 2:
                raise HTTPException(status_code=400, detail="El archivo debe tener dos columnas: wavelength_um y reflectance_%")
            data = data.reshape(1, -1)

        if data.shape[1] != 2:
            raise HTTPException(status_code=400, detail="El archivo debe tener dos columnas: wavelength_um y reflectance_%")

        wavelengths_um = data[:, 0].astype(float)
        reflectance_pct = data[:, 1].astype(float)

        # Validaciones físicas
        if np.any(~np.isfinite(wavelengths_um)) or np.any(~np.isfinite(reflectance_pct)):
            raise HTTPException(status_code=400, detail="El archivo contiene valores no numéricos o infinitos")
        if np.any(wavelengths_um <= 0.0):
            raise HTTPException(status_code=400, detail="Las longitudes de onda deben ser mayores a 0")
        if np.any(reflectance_pct < 0.0) or np.any(reflectance_pct > 100.0):
            raise HTTPException(status_code=400, detail="La reflectancia (%) debe estar entre 0 y 100")

        # 2. Validar temperaturas
        for obj_data in object_data_list:
            if obj_data.temperature <= 0:
                raise HTTPException(status_code=400, detail=f"La temperatura de {obj_data.id} debe ser superior a 0")

        # 3. Guardar archivo de emisividad solo una vez
        hash_value = hashlib.md5(raw).hexdigest()[:16]
        unique_name = f"emissivity_{hash_value}.txt"
        dst_path = os.path.join(OUTPUT_STATIC_DIR, unique_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "wb") as f:
            f.write(raw)

        # 4. Actualizar configuración
        config_scene = get_config_scene_dict()

        for obj_data in object_data_list:
            self.validate_object_id_exists(obj_data.id)

            if obj_data.id not in config_scene["objects"]:
                raise HTTPException(status_code=404, detail=f"Configuración no encontrada para el objeto: {obj_data.id}")

            # Actualizar temperatura y emisividad
            config_scene["objects"][obj_data.id]["temperature"] = obj_data.temperature
            config_scene["objects"][obj_data.id]["emissivity_file"] = dst_path

        save_config_scene_dict(config_scene)

        # 5. Validar rango espectral frente a la cámara
        mensaje = ""
        scene_config = get_config_scene_dict()
        wavelengths_scene = np.array(scene_config.get("wavelengths")) / 1000.0  # µm
        w_min, w_max = np.min(wavelengths_scene), np.max(wavelengths_scene)

        if np.min(wavelengths_um) > w_min and np.max(wavelengths_um) < w_max:
            mensaje = f"Advertencia: El archivo no cubre el rango completo de la cámara ({w_min:.3f}–{w_max:.3f} µm)."
        elif np.min(wavelengths_um) > w_min:
            mensaje = f"Advertencia: Faltan valores menores a {w_min:.3f} µm."
        elif np.max(wavelengths_um) < w_max:
            mensaje = f"Advertencia: Faltan valores mayores a {w_max:.3f} µm."
        else:
            mensaje = "Objetos y archivo de atenuación actualizados correctamente."

        # 6. Actualizar escenas
        obj_ids = [obj_data.id for obj_data in object_data_list]
        scene_service.update_thermal_scene_shapes(object_ids=obj_ids)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()

        return mensaje
    
    @log_execution()
    @handle_file_errors()
    def get_suggested_object_emissivity(self) -> list[str]:
        path = os.fspath(PathManager.get_signatures_dir())
        return [f for f in os.listdir(path) if f.lower().endswith(".txt")]


    def get_object_emissivity_file_by_id(self, object_id: str) -> FileResponse:
        self.validate_object_id_exists(object_id)

        # Construir ruta del archivo de emisividad debajo de OUTPUT_STATIC_DIR
        base_out = OUTPUT_STATIC_DIR
        rel_obj_id = object_id
        if os.path.isabs(object_id):
            rel_obj_id = os.path.splitdrive(object_id)[1].lstrip("\\/")
        dst_no_ext = os.path.splitext(rel_obj_id)[0]
        dst_path = os.path.join(base_out, dst_no_ext + ".txt")

        if not os.path.exists(dst_path):
            raise HTTPException(status_code=404, detail=f"No se encontró archivo de emisividad para el objeto: {object_id}")

        return FileResponse(dst_path, filename=os.path.basename(dst_path))
    
    @log_execution()
    def update_default_emissivity(self, file_name: str, object_id: str) -> None:
        path = os.fspath(PathManager.get_signatures_dir())
        default_file_path = os.path.join(path, file_name)
        
        # Validar que el archivo de emisividad existe
        if not os.path.exists(default_file_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {default_file_path}")
        
        self.validate_object_id_exists(object_id)

        # Construir ruta del archivo de emisividad debajo de OUTPUT_STATIC_DIR
        base_out = OUTPUT_STATIC_DIR
        rel_obj_id = object_id
        if os.path.isabs(object_id):
            rel_obj_id = os.path.splitdrive(object_id)[1].lstrip("\\/")
        dst_no_ext = os.path.splitext(rel_obj_id)[0]
        dst_path = os.path.join(base_out, dst_no_ext + ".txt")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            shutil.copyfile(default_file_path, dst_path)
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error copiando el archivo: {e}"
            )
        
        scene_service.update_thermal_scene_obj(object_id=object_id)
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()
    
    def get_families(self) -> Dict[str, Any]:
        """
        Lista todas las familias de objetos detectadas.
        
        Returns:
            Dict con información de familias
        """
        config = get_config_scene_dict()
        family_mgr = FamilyManager(config)
        
        families = [
            {
                "base_name": f.base_name,
                "members": f.members,
                "count": f.count,
                "shared_properties": f.shared_properties
            }
            for f in family_mgr.list_all_families()
        ]
        
        stats = family_mgr.get_statistics()
        
        return {
            "families": families,
            "statistics": stats,
            "total_families": len(families)
        }
    
    def preview_family_update(self, object_id: str) -> Dict[str, Any]:
        """
        Previsualiza qué objetos se actualizarían en modo familia.
        
        Args:
            object_id: ID del objeto seleccionado
            
        Returns:
            Dict con información de la familia
        """
        config = get_config_scene_dict()
        family_mgr = FamilyManager(config)
        
        return family_mgr.preview_family_update(object_id)
    
    async def update_object_with_mode(
        self,
        object_id: str,
        mode: str,
        temperature: float = None,
        emissivity_file: UploadFile = None,
        is_reflectance: bool = False,
        temp_min: float = None,
        temp_max: float = None
    ) -> Dict[str, Any]:
        """
        Actualiza objeto(s) según modo seleccionado.
        
        Args:
            object_id: ID del objeto seleccionado
            mode: "Objeto" para individual, "Familia" para toda la familia
            temperature: Nueva temperatura en Kelvin (opcional)
            emissivity_file: Nuevo archivo de emisividad (opcional)
            is_reflectance: Si el archivo subido es reflectancia (convertir 1-R)
            temp_min: Temperatura mínima del rango aleatorio (opcional)
            temp_max: Temperatura máxima del rango aleatorio (opcional)
            
        Returns:
            Dict con objects_updated, count, mode y family_name
        """
        if mode not in ["Objeto", "Familia"]:
            raise HTTPException(
                status_code=400,
                detail=f"Modo inválido: '{mode}'. Debe ser 'Objeto' o 'Familia'"
            )
        
        # Validar que el objeto existe físicamente
        self.validate_object_id_exists(object_id)
        
        config = get_config_scene_dict()
        
        # Asegurar que el objeto existe en la configuración (auto-registro si falta)
        if object_id not in config.get("objects", {}):
            logger.info(f"Objeto '{object_id}' no encontrado en config, auto-registrando...")
            self.get_object_info_by_id(object_id)
            config = get_config_scene_dict()
        
        # Preparar propiedades a actualizar
        properties = {}
        if temperature is not None:
            properties["temperature"] = temperature
        
        # Manejar archivo de emisividad si se proporciona
        if emissivity_file is not None and emissivity_file.filename:
            raw = await emissivity_file.read()
            if raw:
                # Si es reflectancia, convertir a emisividad
                if is_reflectance:
                    logger.info("Convirtiendo reflectancia a emisividad (100 - R)")
                    lines = raw.decode("utf-8").splitlines()
                    converted_lines = []
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                wl = parts[0]
                                refl = float(parts[1])
                                emiss = max(0.0, 100.0 - refl)
                                converted_lines.append(f"{wl}\t{emiss:.4f}")
                            except ValueError:
                                converted_lines.append(line)
                        else:
                            converted_lines.append(line)
                    raw = "\n".join(converted_lines).encode("utf-8")

                # Generar nombre único persistente (basado en hash)
                hash_value = hashlib.md5(raw).hexdigest()[:16]
                unique_name = f"emissivity_{hash_value}.txt"
                dst_path = os.path.join(OUTPUT_ASSETS_DIR, unique_name)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with open(dst_path, "wb") as f:
                    f.write(raw)
                properties["emissivity_file"] = str(dst_path)
                logger.info(f"Nuevo archivo de emisividad guardado en {dst_path}")
        
        if not properties and temp_min is None and temp_max is None:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar al menos una propiedad a actualizar o un rango de temperatura"
            )
        
        updated_objects = []
        family_name = None
        
        properties_updated = list(properties.keys())
        if temp_min is not None and temp_max is not None:
            properties_updated.append("temperature")
        
        if mode == "Objeto":
            # Actualizar solo el objeto individual
            if temp_min is not None and temp_max is not None:
                import random
                properties["temperature"] = random.uniform(temp_min, temp_max)
            config["objects"][object_id].update(properties)
            updated_objects = [object_id]
            logger.info(f"Objeto '{object_id}' actualizado")
        
        elif mode == "Familia":
            # Actualizar toda la familia
            family_mgr = FamilyManager(config)
            family = family_mgr.get_family(object_id)
            
            if not family:
                # El objeto no tiene familia, actualizar solo él
                logger.warning(
                    f"Objeto '{object_id}' no pertenece a familia, "
                    f"actualizando solo este objeto"
                )
                if temp_min is not None and temp_max is not None:
                    import random
                    properties["temperature"] = random.uniform(temp_min, temp_max)
                config["objects"][object_id].update(properties)
                updated_objects = [object_id]
            else:
                # Actualizar todos los miembros de la familia
                family_name = family.base_name
                
                # Si se define rango de temperatura, asignar una temperatura aleatoria a cada miembro
                if temp_min is not None and temp_max is not None:
                    import random
                    updated_objects = []
                    for obj_id in family.members:
                        if obj_id in config.get("objects", {}):
                            member_props = dict(properties)
                            member_props["temperature"] = random.uniform(temp_min, temp_max)
                            config["objects"][obj_id].update(member_props)
                            updated_objects.append(obj_id)
                else:
                    # Comportamiento normal (temperatura fija o solo emisividad)
                    updated_objects = family_mgr.update_family(
                        family.base_name,
                        properties
                    )
                logger.info(
                    f"Familia '{family_name}' actualizada: "
                    f"{len(updated_objects)} objetos"
                )
        
        # Guardar cambios
        save_config_scene_dict(config)
        
        # Regenerar escenas térmicas para todos los objetos actualizados
        for obj_id in updated_objects:
            scene_service.update_thermal_scene_obj(object_id=obj_id)
        
        # Preparar escenas auxiliares
        scene_service.prepare_depth_scene()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()
        
        return {
            "objects_updated": updated_objects,
            "count": len(updated_objects),
            "mode": mode.lower(),
            "family_name": family_name,
            "properties_updated": properties_updated
        }
