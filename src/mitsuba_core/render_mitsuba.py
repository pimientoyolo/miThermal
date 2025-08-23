import logging

from src.config import get_output_path, MITSUBA_CONFIG
import os
import json
import numpy as np
from src.mitsuba_core.scene_parser import SceneParser


class RenderRGB:
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        self.mi = mi
        self.scene_parser = SceneParser()
        
        self.logger = logging.getLogger(__name__)

    def render(self, scene_path: str, out_dir: str) -> str:

        scene_dict = self.scene_parser.xml_to_dict(scene_path)

        #spp
        scene_dict['scene']['default'][0]['@value']= int(MITSUBA_CONFIG['spp']/2)

        # guardar ahora como la scena de xml
        with open(scene_path, 'w') as f:
            f.write(self.scene_parser.dict_to_xml(scene_dict))

        # guardar scene_dict como JSON
        json_output_path = os.path.join(out_dir, "scene.json")
        with open(json_output_path, 'w') as json_file:
            json.dump(scene_dict, json_file, indent=2)

        # cargar escena con mitsuba
        scene = self.mi.load_file(scene_path)

        self.logger.info("Inicio de renderizado de escena de muestra")
        image = self.mi.render(scene)
        self.logger.info("Renderizado de escena completado")

        output_path = os.path.join(out_dir, "rendered_image.jpg")
        self.mi.util.write_bitmap(output_path, image)
        return output_path

class RenderDepth():
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        self.mi = mi

    def render(self, scene_path: str, out_dir: str) -> str:
        scene = self.mi.load_file(scene_path)
        image = self.mi.render(scene)
        
        # Convertir la imagen a numpy array
        image_array = np.array(image)
        
        # Extraer solo el canal 0 (primer canal - escala de grises)
        if len(image_array.shape) == 3 and image_array.shape[2] > 1:
            # Si la imagen tiene múltiples canales, tomar el canal 0
            grayscale_channel = image_array[:, :, 0]
        else:
            # Si ya es escala de grises o un solo canal
            grayscale_channel = image_array.squeeze()
        
        # Guardar como archivo numpy
        output_path = os.path.join(out_dir, "depth.npy")
        np.save(output_path, grayscale_channel)
        
        return output_path