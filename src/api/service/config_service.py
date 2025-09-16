import logging
from src.api.dto.cameraDTO import CameraDTO
import src.config as config


logger = logging.getLogger(__name__)

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

    