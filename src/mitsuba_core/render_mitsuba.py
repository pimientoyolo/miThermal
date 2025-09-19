import logging

import os
import json
import numpy as np
from src.mitsuba_core.scene_parser import SceneParser
import src.config as config
from PIL import Image


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

        # Conversión correcta a RGB uint8 sRGB (sin write_bitmap)
        bmp = self.mi.Bitmap(image)
        bmp8 = bmp.convert(self.mi.Bitmap.PixelFormat.RGB,
                           self.mi.Struct.Type.UInt8,
                           srgb_gamma=True)
        arr8 = np.array(bmp8, copy=False)  # [H, W, 3], uint8
        
        img = Image.fromarray(arr8, mode="RGB")
        img.save(config.IMAGE_DIR, format="PNG")

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

        contribution_blackbody_air = np.load(config.CONTRIBUTION_BLACKBODY_AIR_DIR)

        image_array = image_array + contribution_blackbody_air

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

    def render_temperature_map(self):
        scene = self.mi.load_file(config.SCENE_TEMPERATURE_MAP)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # Colapsar bandas: promedio a lo largo del último eje -> [y, x]
        if image_array.ndim == 3:
            image_array = image_array.mean(axis=-1)
        else:
            image_array = image_array.squeeze()

        # crear carpeta si no existe
        os.makedirs(config.OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        np.save(config.TEMPERATURE_MAP_DIR, image_array)

