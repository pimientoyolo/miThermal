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

    def render(self):

        scene = self.mi.load_file(config.SCENE_DEPTH_DIR)

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

        # crear carpeta si no existe
        os.makedirs(config.OUTPUT_STATIC_RESULT_DIR, exist_ok=True)
        
        # Guardar como archivo numpy
        np.save(config.DEPTH_DIR, grayscale_channel)

class RenderThermal():

    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_spectral')
        self.mi = mi

    def render(self):
        scene = self.mi.load_file(config.SCENE_THERMAL_DIR)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # crear carpeta si no existe
        os.makedirs(config.OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        np.save(config.THERMAL_DIR, image_array)

    def render_blackbody_air(self):

        scene = self.mi.load_file(config.SCENE_BLACKBODY_AIR)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # crear carpeta si no existe
        os.makedirs(config.OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        np.save(config.BLACKBODY_AIR_DIR, image_array)

    def render_transmittance_blackbody_air(self):
        scene = self.mi.load_file(config.SCENE_TRANSMITTANCE_BLACKBODY_AIR)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # crear carpeta si no existe
        os.makedirs(config.OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        np.save(config.TRANSMITTANCE_BLACKBODY_AIR_DIR, image_array)

