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

    def render_basic_scene(self):
        """Renderiza escena RGB básica."""
        self._validate_scene_file(SceneType.RGB.value)
        self._render_with_validation(
            SceneType.RGB.value,
            self.render_rgb.render
        )

    def render_depth_image(self):
        """Renderiza mapa de profundidad."""
        self._validate_scene_file(SceneType.DEPTH.value)
        self._render_with_validation(
            SceneType.DEPTH.value,
            self.render_depth.render
        )

    def render_thermal_image(self):
        """Renderiza imagen térmica con calcología de contribución de aire."""
        self.render_contribution_air()
        self._validate_scene_file(SceneType.THERMAL.value)
        self._render_with_validation(
            SceneType.THERMAL.value,
            self.render_thermal.render
        )

    def render_blackbody_air_image(self):
        """Renderiza escena blackbody air."""
        self._validate_scene_file(SceneType.BLACKBODY_AIR.value)
        self._render_with_validation(
            SceneType.BLACKBODY_AIR.value,
            self.render_thermal.render_blackbody_air
        )

    def render_transmittance_blackbody_air_image(self):
        """Renderiza escena transmittance blackbody air."""
        self._validate_scene_file(SceneType.TRANSMITTANCE_BLACKBODY_AIR.value)
        self._render_with_validation(
            SceneType.TRANSMITTANCE_BLACKBODY_AIR.value,
            self.render_thermal.render_transmittance_blackbody_air
        )

    def render_contribution_air(self):
        """
        Calcula la contribución de radiancia del aire (path radiance) analíticamente
        usando el mapa de profundidad y la ley de Beer-Lambert.
        L_path = L_air * (1 - exp(-sigma_t * depth))
        """
        from src.config import get_config_scene_dict
        from src.atmosphere import get_gas_manager
        from src.utils.objects.objects import ObjectUtils
        
        # 1. Asegurar que tenemos el mapa de profundidad
        self.render_depth_image()
        depth_path = self.path_manager.get_result_path(SceneType.DEPTH.value)
        depth = load_numpy_array(depth_path) # [H, W]
        
        # Manejar fondo (Mitsuba depth suele retornar 0 o valores muy grandes para el infinito)
        # Si es 0, lo tratamos como infinito para que tenga radiancia de aire completa
        depth[depth == 0] = 1e6 

        # 2. Obtener datos de atmósfera y cámara
        config_scene = get_config_scene_dict()
        t_air = config_scene["air"]["temperature"]
        wavelengths_scene = np.array(config_scene["wavelengths"]) # nm
        
        # Leer sigma_t (ya corregido a m^-1)
        wl_air, sigma_t_air = get_gas_manager().load_gas("air")
        
        # Interpolar sigma_t a las longitudes de onda de la cámara
        sigma_t_interp = np.interp(wavelengths_scene, wl_air, sigma_t_air)
        
        # 3. Calcular radiancia de cuerpo negro del aire para cada banda
        obj_utils = ObjectUtils()
        l_air_spectral = obj_utils.blackbody_radiance_nm(wavelengths_scene, t_air) # [Bands]
        
        # 4. Calcular contribución (H, W, Bands)
        # Expandir dimensiones para broadcasting: depth[H,W,1] * sigma[1,1,B]
        depth_exp = depth[:, :, np.newaxis]
        sigma_exp = sigma_t_interp[np.newaxis, np.newaxis, :]
        l_air_exp = l_air_spectral[np.newaxis, np.newaxis, :]
        
        # Transmitancia: tau = exp(-sigma * d)
        transmittance = np.exp(-sigma_exp * depth_exp)
        
        # Radiancia de camino: L_path = L_air * (1 - tau)
        contribution = l_air_exp * (1.0 - transmittance)
        
        # 4.5 Recortar bandas de seguridad (primera y última)
        if contribution.shape[-1] > 2:
            contribution = contribution[:, :, 1:-1]
            logger.info(f"Bandas de seguridad recortadas de la contribución de aire. Nueva forma: {contribution.shape}")

        # 5. Guardar resultado
        contribution_path = self.path_manager.get_result_path("contribution_blackbody_air")
        save_numpy_array(contribution_path, contribution)
        logger.info(f"Contribución de aire calculada analíticamente usando depth map")

    def render_temperature_map(self):
        """Renderiza mapa de temperatura."""
        self._validate_scene_file(SceneType.TEMPERATURE_MAP.value)
        self._render_with_validation(
            SceneType.TEMPERATURE_MAP.value,
            self.render_thermal.render_temperature_map
        )

    def render_emissivity_map(self):
        """Renderiza mapa de emisividad integrada."""
        self._validate_scene_file(SceneType.EMISSIVITY_MAP.value)
        self._render_with_validation(
            SceneType.EMISSIVITY_MAP.value,
            self.render_thermal.render_emissivity_map
        )
