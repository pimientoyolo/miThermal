import logging
import io
import os
from src.api.dto.cameraDTO import CameraDTO, UpdateCameraDTO
from src.api.service.scene_service import SceneService
from src.config import PathManager, DEFAULT_ATTENNUATION_DIR, get_config_scene_dict, save_config_scene_dict
import numpy as np

from fastapi import HTTPException, UploadFile

from src.utils.objects.objects import ObjectUtils
from src.utils.decorators import log_execution, handle_file_errors


logger = logging.getLogger(__name__)

scene_service = SceneService()
object_utils = ObjectUtils()
path_manager = PathManager

class ConfigService:
    def __init__(self):
        self.logger = logger

    @log_execution()
    def get_camera_config(self) -> CameraDTO:
        
        scene_config = get_config_scene_dict()
        camera_data = scene_config.get("camera", {})

        # Obtener valores básicos
        spp = camera_data.get("spp", 256)
        width = camera_data.get("width", 256)
        height = camera_data.get("height", 256)
        fov = camera_data.get("fov", 45.0)
        
        rotate_x = camera_data.get("rotate_x", 0.0)
        rotate_y = camera_data.get("rotate_y", 0.0)
        rotate_z = camera_data.get("rotate_z", 0.0)
        
        translate_x = camera_data.get("translate_x", 0.0)
        translate_y = camera_data.get("translate_y", 0.0)
        translate_z = camera_data.get("translate_z", 0.0)

        # Obtener valores esféricos si existen, o calcularlos (opcional)
        theta = camera_data.get("theta")
        phi = camera_data.get("phi")
        radius = camera_data.get("radius")
        target_x = camera_data.get("target_x", 0.0)
        target_y = camera_data.get("target_y", 0.0)
        target_z = camera_data.get("target_z", 0.0)

        camera_dto = CameraDTO(
            spp=spp,
            width=width,
            height=height,
            wavelengths=[w / 1000 for w in scene_config.get("wavelengths", [])],
            num_bands=scene_config.get("num_bands", 0),
            rotate_x=rotate_x,
            rotate_y=rotate_y,
            rotate_z=rotate_z,
            translate_x=translate_x,
            translate_y=translate_y,
            translate_z=translate_z,
            fov=fov,
            theta=theta,
            phi=phi,
            radius=radius,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z
        )

        return camera_dto

    @log_execution()
    def update_camera_config(self, camera_update: UpdateCameraDTO) -> CameraDTO:

        scene_config = get_config_scene_dict()
        camera_data = scene_config.setdefault("camera", {})

        # Actualizar valores comunes
        camera_data["spp"] = camera_update.spp
        camera_data["width"] = camera_update.width
        camera_data["height"] = camera_update.height
        camera_data["fov"] = camera_update.fov

        # Si se proporcionan coordenadas esféricas, calcular cartesianas
        if camera_update.theta is not None and camera_update.phi is not None and camera_update.radius is not None:
            from src.api.service.scene_service import calculate_look_at_blender
            
            theta_rad = np.deg2rad(camera_update.theta)
            phi_rad = np.deg2rad(camera_update.phi)
            radius = camera_update.radius
            
            target_x = camera_update.target_x if camera_update.target_x is not None else camera_data.get("target_x", 0.0)
            target_y = camera_update.target_y if camera_update.target_y is not None else camera_data.get("target_y", 0.0)
            target_z = camera_update.target_z if camera_update.target_z is not None else camera_data.get("target_z", 0.0)
            
            # Calcular posición relativa
            x = radius * np.sin(theta_rad) * np.cos(phi_rad)
            y = radius * np.sin(theta_rad) * np.sin(phi_rad)
            z = radius * np.cos(theta_rad)
            
            # Posición absoluta
            translate_x = target_x + x
            translate_y = target_y + y
            translate_z = target_z + z
            
            # Calcular rotación look-at
            cam_pos = np.array([translate_x, translate_y, translate_z])
            target_pos = np.array([target_x, target_y, target_z])
            rx, ry, rz = calculate_look_at_blender(cam_pos, target_pos)
            
            # Guardar en config
            camera_data["translate_x"] = float(translate_x)
            camera_data["translate_y"] = float(translate_y)
            camera_data["translate_z"] = float(translate_z)
            camera_data["rotate_x"] = rx
            camera_data["rotate_y"] = ry
            camera_data["rotate_z"] = rz
            
            # Guardar esféricas también
            camera_data["theta"] = camera_update.theta
            camera_data["phi"] = camera_update.phi
            camera_data["radius"] = radius
            camera_data["target_x"] = target_x
            camera_data["target_y"] = target_y
            camera_data["target_z"] = target_z
            
        else:
            # Usar cartesianas si se proporcionan, o mantener actuales
            if camera_update.rotate_x is not None: camera_data["rotate_x"] = camera_update.rotate_x
            if camera_update.rotate_y is not None: camera_data["rotate_y"] = camera_update.rotate_y
            if camera_update.rotate_z is not None: camera_data["rotate_z"] = camera_update.rotate_z
            if camera_update.translate_x is not None: camera_data["translate_x"] = camera_update.translate_x
            if camera_update.translate_y is not None: camera_data["translate_y"] = camera_update.translate_y
            if camera_update.translate_z is not None: camera_data["translate_z"] = camera_update.translate_z
            
            # Limpiar esféricas si se movió manualmente por cartesianas (opcional)
            # o podríamos intentar re-calcularlas aquí.

        # Validaciones
        if camera_update.fov < 1 or camera_update.fov > 179:
            raise HTTPException(status_code=400, detail="FOV debe estar entre 1 y 179 grados")
        if camera_update.spp <= 0:
            raise HTTPException(status_code=400, detail="SPP debe ser mayor a 0")
        if (camera_update.spp & (camera_update.spp - 1)) != 0:
            raise HTTPException(status_code=400, detail="SPP debe ser potencia de 2")
        if camera_update.width <= 0 or camera_update.height <= 0:
            raise HTTPException(status_code=400, detail="Alto y ancho deben ser mayores a 0")

        save_config_scene_dict(scene_config)

        scene_service.update_scene_camera_all()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()

        return self.get_camera_config()

    @log_execution()
    def export_camera_spatial_config(self) -> dict:
        """Exporta la configuración espacial de la cámara"""
        config = get_config_scene_dict()
        cam = config.get("camera", {})
        return {
            "rotate_x": cam.get("rotate_x", 0.0),
            "rotate_y": cam.get("rotate_y", 0.0),
            "rotate_z": cam.get("rotate_z", 0.0),
            "translate_x": cam.get("translate_x", 0.0),
            "translate_y": cam.get("translate_y", 0.0),
            "translate_z": cam.get("translate_z", 0.0),
            "fov": cam.get("fov", 45.0),
            "theta": cam.get("theta"),
            "phi": cam.get("phi"),
            "radius": cam.get("radius"),
            "target_x": cam.get("target_x"),
            "target_y": cam.get("target_y"),
            "target_z": cam.get("target_z")
        }

    @log_execution()
    def import_camera_spatial_config(self, spatial_config: dict) -> CameraDTO:
        """Importa la configuración espacial de la cámara"""
        scene_config = get_config_scene_dict()
        cam = scene_config.setdefault("camera", {})
        
        # Solo actualizar campos espaciales
        spatial_fields = [
            "rotate_x", "rotate_y", "rotate_z", 
            "translate_x", "translate_y", "translate_z", 
            "fov", "theta", "phi", "radius",
            "target_x", "target_y", "target_z"
        ]
        
        for field in spatial_fields:
            if field in spatial_config and spatial_config[field] is not None:
                cam[field] = spatial_config[field]
        
        save_config_scene_dict(scene_config)
        
        scene_service.update_scene_camera_all()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()
        
        return self.get_camera_config()

    @log_execution()
    def update_wavelengths(self, w_min: int, w_max: int, bands: int) -> CameraDTO:

        # validar que w_max sea mayor que w_min
        if w_max < w_min:
            raise HTTPException(status_code=400, detail="El maximo debe ser mayor que el minimo")

        scene_config = get_config_scene_dict()

        wavelengths = np.linspace(w_min, w_max, bands, endpoint=True, dtype=int).tolist()

        scene_config["wavelengths"] = wavelengths
        scene_config["num_bands"] = bands

        save_config_scene_dict(scene_config)

        update = scene_service.update_film_spectrum()

        if update:
            scene_service.prepare_blackbody_air_scene()
            scene_service.prepare_depth_scene()
            scene_service.prepare_transmittance_blackbody_air_scene()
            scene_service.prepare_temperature_map()

        return self.get_camera_config()
    
    @log_execution()
    def get_air_temperature(self) -> float:
        scene_config = get_config_scene_dict()
        return scene_config["air"]["temperature"]
    
    @log_execution()
    def set_air_temperature(self, temperature: float):

        # validar temperatura mayor a 0
        if temperature <= 0:
            raise HTTPException(status_code=400, detail="La temperatura debe ser superior a 0 (cero absoluto no permitido)")

        scene_config = get_config_scene_dict()

        scene_config["air"]["temperature"] = temperature

        save_config_scene_dict(scene_config)

        scene_service.update_thermal_scene_air()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_temperature_map()

    @log_execution()
    @handle_file_errors()
    def set_air_attenuation(self, file: UploadFile) -> str:

        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No se envio archivo")

        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="El archivo debe ser .txt")

        # Nota: Por ahora no guardamos ni comparamos con wavelengths de config hasta validar todo correctamente

        scene_config = get_config_scene_dict()

        wavelengths_nm = np.array(scene_config["wavelengths"], dtype=int)
        w_min = wavelengths_nm.min() / 1000.0  # um
        w_max = wavelengths_nm.max() / 1000.0  # um

        # Leer el archivo y validar su contenido (TAB-delimited, 2 columnas)
        try:
            raw = file.file.read()
            if not raw or len(raw) == 0:
                raise HTTPException(status_code=400, detail="El archivo está vacío")

            wavelengths_um, sigma_t = self._parse_air_attenuation_tsv(raw)

            if wavelengths_um.size == 0 or sigma_t.size == 0:
                raise HTTPException(status_code=400, detail="El archivo no contiene datos válidos")
            
            if wavelengths_um.size != sigma_t.size:
                raise HTTPException(status_code=400, detail="El archivo debe tener el mismo número de valores en ambas columnas")
            
            # Guardar el archivo en la ubicación configurada
            air_attenuation_path = path_manager.get_air_attenuation_path()
            with open(air_attenuation_path, "wb") as f:
                f.write(raw)

            mensaje = ""

            if np.min(wavelengths_um) > w_min and np.max(wavelengths_um) < w_max:
                mensaje =  "Advertencia: El archivo contiene longitudes de onda que no cubre el rango completo de la cámara. Se recomienda incluir valores entre {:.3f} µm y {:.3f} µm".format(w_min, w_max)
            
            elif np.min(wavelengths_um) > w_min:
                mensaje =  "Advertencia: El archivo contiene longitudes de onda mayores al mínimo de la cámara ({:.3f} µm). Se recomienda incluir valores menores".format(w_min)

            elif np.max(wavelengths_um) < w_max:
                mensaje =  "Advertencia: El archivo contiene longitudes de onda menores al máximo de la cámara ({:.3f} µm). Se recomienda incluir valores mayores".format(w_max)
            
            else:
                mensaje = "Archivo de atenuación del aire actualizado correctamente"  

            scene_service.update_thermal_scene_air()
            scene_service.prepare_blackbody_air_scene()
            scene_service.prepare_transmittance_blackbody_air_scene()
            scene_service.prepare_depth_scene()
            scene_service.prepare_temperature_map()

            return mensaje

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error validando archivo de atenuación: {e}")
            raise HTTPException(status_code=500, detail="Error interno al validar archivo de atenuación")

    def _parse_air_attenuation_tsv(self, raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """
        Parsea y valida el contenido del archivo de atenuación del aire:
        - Debe estar separado por TABs
        - Debe tener exactamente 2 columnas: wavelength_um (>0) y sigma_t (>=0)
        - Valores deben ser finitos
        Retorna (wavelengths_um, sigma_t)
        """
        try:
            # Forzar lectura por TABs (UTF-8)
            try:
                data = np.loadtxt(io.StringIO(raw_bytes.decode('utf-8')), delimiter='\t')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error leyendo el archivo. Debe estar separado por TABs: {e}")

            # Normalizar a 2D si viene en una sola fila
            if data.ndim == 1:
                if data.size < 2:
                    raise HTTPException(status_code=400, detail="El archivo debe tener exactamente dos columnas: wavelength_um y sigma_t")
                data = data.reshape(1, -1)

            # Debe tener exactamente dos columnas
            if data.shape[1] != 2:
                raise HTTPException(status_code=400, detail="El archivo debe tener exactamente dos columnas: wavelength_um y sigma_t")

            wavelengths_um = data[:, 0].astype(float)
            sigma_t = data[:, 1].astype(float)

            # Validar NaNs o infinitos
            if np.any(~np.isfinite(wavelengths_um)) or np.any(~np.isfinite(sigma_t)):
                raise HTTPException(status_code=400, detail="El archivo contiene valores no numéricos o infinitos")

            # Reglas de dominio
            if np.any(wavelengths_um <= 0):
                raise HTTPException(status_code=400, detail="Las longitudes de onda (µm) deben ser mayores a 0")
            if np.any(sigma_t < 0):
                raise HTTPException(status_code=400, detail="Los valores de atenuación (sigma_t) no pueden ser negativos")

            return wavelengths_um, sigma_t
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error parseando archivo de atenuación: {e}")
            raise HTTPException(status_code=500, detail="Error interno al parsear archivo de atenuación")
    
    def suggest_air_attenuation(self) -> list[str]:

        path = os.fspath(DEFAULT_ATTENNUATION_DIR)

        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Directorio no encontrado: {path}")
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"No es un directorio: {path}")

        try:
            return [f for f in os.listdir(path) if f.lower().endswith(".txt")]
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Error leyendo el directorio: {e}")
        
    def set_air_attenuation_by_filename(self, file_name: str) -> str:

        path = os.fspath(DEFAULT_ATTENNUATION_DIR)

        # Validaciones de entrada
        if not file_name:
            raise HTTPException(status_code=400, detail="Debe proporcionar un nombre de archivo")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Directorio no encontrado: {path}")
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"No es un directorio: {path}")

        # Evitar path traversal y construir ruta completa
        base_name = os.path.basename(file_name)
        if not base_name.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="El archivo debe ser .txt")

        src_file = os.path.join(path, base_name)

        logger.info(f"Intentando establecer atenuación del aire desde archivo: {src_file}")

        if not os.path.exists(src_file) or not os.path.isfile(src_file):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {src_file}")

        wavelengths_nm, sigma_t = object_utils.get_attenuation(file_name)

        data = np.column_stack((wavelengths_nm / 1000.0, sigma_t))  # Convertir nm a um

        air_attenuation_path = path_manager.get_air_attenuation_path()
        np.savetxt(air_attenuation_path, data, delimiter='\t')

        scene_service.update_thermal_scene_air()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_temperature_map()

        
        

