import logging
from src.api.dto.cameraDTO import CameraDTO, UpdateCameraDTO
from src.api.service.scene_service import SceneService
import src.config as config
import numpy as np

from fastapi import HTTPException


logger = logging.getLogger(__name__)

scene_service = SceneService()

class ConfigService:
    def __init__(self):
        self.logger = logger

    def get_camera_config(self) -> CameraDTO:
        
        scene_config = config.get_config_scene_dict()

        camera_dto = CameraDTO(
            spp=scene_config["camera"]["spp"],
            width=scene_config["camera"]["width"],
            height=scene_config["camera"]["height"],
            wavelengths=scene_config["wavelengths"],
            num_bands=scene_config["num_bands"]
        )

        return camera_dto

    def update_camera_config(self, camera_update: UpdateCameraDTO) -> CameraDTO:

        scene_config = config.get_config_scene_dict()

        scene_config["camera"]["spp"] = camera_update.spp
        scene_config["camera"]["width"] = camera_update.width
        scene_config["camera"]["height"] = camera_update.height

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
            wavelengths=scene_config["wavelengths"],
            num_bands=scene_config["num_bands"]
        )

        config.save_config_scene_dict(scene_config)

        scene_service.update_scene_camera_rgb()
        scene_service.update_scene_camera_thermal()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_depth_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()


        return updated_camera

    def update_wavelengths(self, w_min: int, w_max: int, bands: int) -> CameraDTO:

        # validar que w_max sea mayor que w_min
        if w_max < w_min:
            raise HTTPException(status_code=400, detail="El maximo debe ser mayor que el minimo")

        scene_config = config.get_config_scene_dict()

        wavelengths = np.linspace(w_min, w_max, bands, endpoint=True, dtype=int).tolist()

        scene_config["wavelengths"] = wavelengths
        scene_config["num_bands"] = bands

        config.save_config_scene_dict(scene_config)

        update = scene_service.update_film_spectrum()

        if update:
            scene_service.prepare_blackbody_air_scene()
            scene_service.prepare_depth_scene()
            scene_service.prepare_transmittance_blackbody_air_scene()

        return self.get_camera_config()
    
    def get_air_temperature(self) -> float:
        scene_config = config.get_config_scene_dict()
        return scene_config["air"]["temperature"]
    
    def set_air_temperature(self, temperature: float):

        # validar temperatura mayor a 0
        if temperature <= 0:
            raise HTTPException(status_code=400, detail="La temperatura debe ser superior a 0 (cero absoluto no permitido)")

        scene_config = config.get_config_scene_dict()

        scene_config["air"]["temperature"] = temperature

        config.save_config_scene_dict(scene_config)

        scene_service.update_thermal_scene_air()
        scene_service.prepare_blackbody_air_scene()
        scene_service.prepare_transmittance_blackbody_air_scene()
        scene_service.prepare_depth_scene()