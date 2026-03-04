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

        camera_dto = CameraDTO(
            spp=scene_config["camera"]["spp"],
            width=scene_config["camera"]["width"],
            height=scene_config["camera"]["height"],
            wavelengths=[w / 1000 for w in scene_config["wavelengths"]],
            num_bands=scene_config["num_bands"],
            rotate_x=scene_config["camera"].get("rotate_x", 0.0),
            rotate_y=scene_config["camera"].get("rotate_y", 0.0),
            rotate_z=scene_config["camera"].get("rotate_z", 0.0),
            translate_x=scene_config["camera"].get("translate_x", 0.0),
            translate_y=scene_config["camera"].get("translate_y", 0.0),
            translate_z=scene_config["camera"].get("translate_z", 0.0),
            fov=scene_config["camera"].get("fov", 45.0)
        )

        return camera_dto

    @log_execution()
    def update_camera_config(self, camera_update: UpdateCameraDTO) -> CameraDTO:

        scene_config = get_config_scene_dict()

        scene_config["camera"]["spp"] = camera_update.spp
        scene_config["camera"]["width"] = camera_update.width
        scene_config["camera"]["height"] = camera_update.height
        scene_config["camera"]["rotate_x"] = camera_update.rotate_x
        scene_config["camera"]["rotate_y"] = camera_update.rotate_y
        scene_config["camera"]["rotate_z"] = camera_update.rotate_z
        scene_config["camera"]["translate_x"] = camera_update.translate_x
        scene_config["camera"]["translate_y"] = camera_update.translate_y
        scene_config["camera"]["translate_z"] = camera_update.translate_z
        scene_config["camera"]["fov"] = camera_update.fov

        #validar fov entre 1 y 179
        if camera_update.fov < 1 or camera_update.fov > 179:
            raise HTTPException(status_code=400, detail="FOV debe estar entre 1 y 179 grados")

        # validar spp mayor a 0
        if camera_update.spp <= 0:
            raise HTTPException(status_code=400, detail="SPP debe ser mayor a 0")
        
        # validar que spp sea potencia de 2
        if (camera_update.spp & (camera_update.spp - 1)) != 0:
            raise HTTPException(status_code=400, detail="SPP debe ser potencia de 2")
        
        # validar width y height mayor a 0
        if camera_update.width <= 0 or camera_update.height <= 0:
            raise HTTPException(status_code=400, detail="Alto y ancho deben ser mayores a 0")

        updated_camera = CameraDTO(
            spp=scene_config["camera"]["spp"],
            width=scene_config["camera"]["width"],
            height=scene_config["camera"]["height"],
            wavelengths=[w / 1000 for w in scene_config["wavelengths"]],
            num_bands=scene_config["num_bands"],
            rotate_x=scene_config["camera"]["rotate_x"],
            rotate_y=scene_config["camera"]["rotate_y"],
            rotate_z=scene_config["camera"]["rotate_z"],
            translate_x=scene_config["camera"]["translate_x"],
            translate_y=scene_config["camera"]["translate_y"],
            translate_z=scene_config["camera"]["translate_z"],
            fov=scene_config["camera"]["fov"]
        )

        save_config_scene_dict(scene_config)

        scene_service.update_scene_camera_rgb()
        scene_service.update_scene_camera_thermal()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_temperature_map()

        return updated_camera

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

        
        

