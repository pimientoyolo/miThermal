import logging
import numpy as np

from src.mitsuba_core.render_mitsuba import RenderRGB, RenderDepth, RenderThermal
from src.config import PathManager
from src.api.service.base_services import RenderServiceBase
from src.api.service.scene_helpers import SceneType
from src.utils.helpers import load_numpy_array, save_numpy_array
from src.utils.decorators import log_execution

logger = logging.getLogger(__name__)

class RenderService(RenderServiceBase):
    
    def __init__(self):
        super().__init__()
        self.render_rgb = RenderRGB()
        self.render_depth = RenderDepth()
        self.render_thermal = RenderThermal()
        self.path_manager = PathManager

    def _validate_scene_file(self, scene_type: str) -> None:
        """
        Valida que el archivo de escena XML exista antes de renderizar.
        
        Args:
            scene_type: Tipo de escena (usar SceneType enum)
        """
        scene_path = self.path_manager.get_scene_path(scene_type)
        self._validate_file_exists(scene_path, f"escena {scene_type}")
        self._validate_is_file(scene_path)

    @log_execution()
    def render_basic_scene(self):
        """Renderiza escena RGB básica."""
        self._validate_scene_file(SceneType.RGB.value)
        self._render_with_validation(
            SceneType.RGB.value,
            self.render_rgb.render
        )

    @log_execution()
    def render_depth_image(self):
        """Renderiza mapa de profundidad."""
        self._validate_scene_file(SceneType.DEPTH.value)
        self._render_with_validation(
            SceneType.DEPTH.value,
            self.render_depth.render
        )

    @log_execution()
    def render_thermal_image(self):
        """Renderiza imagen térmica con calcología de contribución de aire."""
        self.render_contribution_air()
        self._validate_scene_file(SceneType.THERMAL.value)
        self._render_with_validation(
            SceneType.THERMAL.value,
            self.render_thermal.render
        )

    @log_execution()
    def render_blackbody_air_image(self):
        """Renderiza escena blackbody air."""
        self._validate_scene_file(SceneType.BLACKBODY_AIR.value)
        self._render_with_validation(
            SceneType.BLACKBODY_AIR.value,
            self.render_thermal.render_blackbody_air
        )

    @log_execution()
    def render_transmittance_blackbody_air_image(self):
        """Renderiza escena transmittance blackbody air."""
        self._validate_scene_file(SceneType.TRANSMITTANCE_BLACKBODY_AIR.value)
        self._render_with_validation(
            SceneType.TRANSMITTANCE_BLACKBODY_AIR.value,
            self.render_thermal.render_transmittance_blackbody_air
        )

    @log_execution()
    def render_contribution_air(self):
        """Calcula contribución de aire restando transmittance de blackbody."""
        self.render_blackbody_air_image()
        self.render_transmittance_blackbody_air_image()
        
        blackbody_path = self.path_manager.get_result_path(SceneType.BLACKBODY_AIR.value)
        transmittance_path = self.path_manager.get_result_path(SceneType.TRANSMITTANCE_BLACKBODY_AIR.value)
        contribution_path = self.path_manager.get_result_path("contribution_blackbody_air")
        
        # Usar helpers para manejo robusto de archivos numpy
        blackbody_air = load_numpy_array(blackbody_path)
        transmittance_blackbody_air = load_numpy_array(transmittance_path)
        contribution_blackbody_air = blackbody_air - transmittance_blackbody_air
        save_numpy_array(contribution_path, contribution_blackbody_air)

    @log_execution()
    @log_execution()
    def render_temperature_map(self):
        """Renderiza mapa de temperatura."""
        self._validate_scene_file(SceneType.TEMPERATURE_MAP.value)
        self._render_with_validation(
            SceneType.TEMPERATURE_MAP.value,
            self.render_thermal.render_temperature_map
        )
