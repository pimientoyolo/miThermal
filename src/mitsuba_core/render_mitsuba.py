import logging

from src.config import get_output_path, MITSUBA_CONFIG
import os
import xmltodict
import json


class RenderRGB:
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        self.mi = mi
        
        self.logger = logging.getLogger(__name__)

    def render(self, scene_path: str, out_dir: str) -> str:

        with open(scene_path) as f:
            scene_dict = xmltodict.parse(f.read())

        #spp
        scene_dict['scene']['default'][0]['@value']= int(MITSUBA_CONFIG['spp']/2)

        # guardar ahora como la scena de xml
        with open(scene_path, 'w') as f:
            f.write(xmltodict.unparse(scene_dict, pretty=True))

        # cargar escena con mitsuba
        scene = self.mi.load_file(scene_path)

        self.logger.info("Inicio de renderizado de escena de muestra")
        image = self.mi.render(scene)
        self.logger.info("Renderizado de escena completado")

        output_path = os.path.join(out_dir, "rendered_image.jpg")
        self.mi.util.write_bitmap(output_path, image)
        return output_path