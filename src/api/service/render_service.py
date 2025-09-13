import logging
import os
from typing import Dict, Any, Optional
from fastapi import HTTPException

from src.mitsuba_core.scenes import Scene
from src.mitsuba_core.render_mitsuba import RenderRGB, RenderDepth, RenderThermal
import src.config as config

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
        self.render_rgb.render()

    def render_depth_image(self, path_xml_scene: str) -> str:
        self._validate_scene_file(path_xml_scene, "depth")
        path_image = self.render_depth.render(scene_path=path_xml_scene, out_dir=self.output_path)
        return path_image

    def render_thermal_image(self, path_xml_scene: str) -> str:
        self._validate_scene_file(path_xml_scene, "thermal")
        path_image = self.render_thermal.render(scene_path=path_xml_scene, out_dir=self.output_path)
        return path_image