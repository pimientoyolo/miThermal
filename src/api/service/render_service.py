import logging
import os
from fastapi import HTTPException

from src.mitsuba_core.render_mitsuba import RenderRGB, RenderDepth, RenderThermal
import src.config as config
import numpy as np

logger = logging.getLogger(__name__)

class RenderService:
    def __init__(self):
        self.logger = logger
        self.render_rgb = RenderRGB()
        self.render_depth = RenderDepth()
        self.render_thermal = RenderThermal()
        self.output_path = config.OUTPUT_DIR

    def _validate_scene_file(self, scene_path: str, scene_type: str) -> None:
        """
        Valida que el archivo de escena XML exista antes de renderizar.
        
        Args:
            scene_path: Ruta del archivo XML de la escena
            scene_type: Tipo de escena para el mensaje de error (RGB, depth, thermal, etc.)
            
        Raises:
            HTTPException: Si el archivo no existe o no es válido
        """
        if not os.path.exists(scene_path):
            raise HTTPException(
                status_code=404, 
                detail=f"El archivo de escena {scene_type} no existe: {scene_path}. "
                       f"Asegúrate de haber preparado la escena {scene_type} primero."
            )
        
        if not os.path.isfile(scene_path):
            raise HTTPException(
                status_code=400, 
                detail=f"La ruta especificada no es un archivo válido: {scene_path}"
            )

    def render_basic_scene(self):
        self._validate_scene_file(config.SCENE_DIR, "RGB")
        try:
            self.render_rgb.render()
        except Exception as e:
            self.logger.error(f"Error al renderizar la escena RGB: {e}")
            raise HTTPException(status_code=500, detail="Error al renderizar: prueba bajar spp y resolución")

    def render_depth_image(self):
        self._validate_scene_file(config.SCENE_DEPTH_DIR, "depth")
        try:
            self.render_depth.render()
        except Exception as e:
            self.logger.error(f"Error al renderizar la escena de profundidad: {e}")
            raise HTTPException(status_code=500, detail="Error al renderizar: prueba bajar spp y resolución")

    def render_thermal_image(self):
        self._validate_scene_file(config.SCENE_THERMAL_DIR, "thermal")
        self.render_contribution_air()
        try:
            self.render_thermal.render()
        except Exception as e:
            self.logger.error(f"Error al renderizar la escena térmica: {e}")
            raise HTTPException(status_code=500, detail="Error al renderizar: prueba bajar spp y resolución")

    def render_blackbody_air_image(self):
        self._validate_scene_file(config.SCENE_BLACKBODY_AIR, "blackbody air")
        try:
            self.render_thermal.render_blackbody_air()
        except Exception as e:
            self.logger.error(f"Error al renderizar la escena de blackbody air: {e}")
            raise HTTPException(status_code=500, detail="Error al renderizar: prueba bajar spp y resolución")

    def render_transmittance_blackbody_air_image(self):
        self._validate_scene_file(config.SCENE_TRANSMITTANCE_BLACKBODY_AIR, "transmittance blackbody air")
        try:
            self.render_thermal.render_transmittance_blackbody_air()
        except Exception as e:
            self.logger.error(f"Error al renderizar la escena de transmittance blackbody air: {e}")
            raise HTTPException(status_code=500, detail="Error al renderizar: prueba bajar spp y resolución")

    def render_contribution_air(self):
        self.render_blackbody_air_image()
        self.render_transmittance_blackbody_air_image()
        blackbody_air = np.load(config.BLACKBODY_AIR_DIR)
        transmittance_blackbody_air = np.load(config.TRANSMITTANCE_BLACKBODY_AIR_DIR)

        contribution_blackbody_air = blackbody_air - transmittance_blackbody_air
        np.save(config.CONTRIBUTION_BLACKBODY_AIR_DIR, contribution_blackbody_air)

    def render_temperature_map(self):
        self._validate_scene_file(config.SCENE_TEMPERATURE_MAP, "temperature map")
        try:
            self.render_thermal.render_temperature_map()
        except Exception as e:
            self.logger.error(f"Error al renderizar la escena del mapa de temperatura: {e}")
            raise HTTPException(status_code=500, detail="Error al renderizar: prueba bajar spp y resolución")
