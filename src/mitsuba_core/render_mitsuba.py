import logging

import os
import json
import numpy as np
from src.mitsuba_core.scene_parser import SceneParser
import src.config as config


class RenderRGB:
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        self.mi = mi
        self.scene_parser = SceneParser()
        
        self.logger = logging.getLogger(__name__)

    def render(self) -> str:

        # cargar escena con mitsuba
        scene = self.mi.load_file(config.SCENE_DIR)

        # renderizar imagen
        image = self.mi.render(scene)

        # crear carpeta si no existe
        os.makedirs(config.OUTPUT_STATIC_RESULT_DIR, exist_ok=True)
        
        # guardar imagen
        self.mi.util.write_bitmap(config.IMAGE_DIR, image)

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

class RenderThermal():

    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_spectral')
        self.mi = mi

    def render(self, scene_path: str, out_dir: str) -> str:
        scene = self.mi.load_file(scene_path)
        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # Guardar toda la información (todos los canales)
        output_path = os.path.join(out_dir, "thermal.npy")
        np.save(output_path, image_array)

        return output_path
