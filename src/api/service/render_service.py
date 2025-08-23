import logging
from typing import Dict, Any, Optional

from src.mitsuba_core.scenes import Scene
from src.mitsuba_core.render_mitsuba import RenderRGB, RenderDepth
from src.config import get_output_path

logger = logging.getLogger(__name__)

class RenderService:
    def __init__(self):
        self.logger = logger
        self.render_rgb = RenderRGB()
        self.render_depth = RenderDepth()
        self.output_path = get_output_path("static")

    def render_basic_scene(self, path_xml_scene: str) -> str:
        path_image = self.render_rgb.render(scene_path=path_xml_scene, out_dir=self.output_path)
        return path_image

    def render_depth_image(self, path_xml_scene: str) -> str:
        path_image = self.render_depth.render(scene_path=path_xml_scene, out_dir=self.output_path)
        return path_image