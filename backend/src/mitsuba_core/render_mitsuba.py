import logging
import os
import json
import numpy as np
from src.utils.scene.parser import SceneParser
from src.config import PathManager, OUTPUT_STATIC_RESULT_DIR
from PIL import Image


class RenderRGB:
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        self.mi = mi
        self.scene_parser = SceneParser()
        self.path_manager = PathManager
        self.logger = logging.getLogger(__name__)

    def render(self) -> str:
        # Cargar escena con mitsuba
        scene_path = self.path_manager.get_scene_path("rgb")
        scene = self.mi.load_file(scene_path)

        # Renderizar imagen
        image = self.mi.render(scene)

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Conversión correcta a RGB uint8 sRGB (sin write_bitmap)
        bmp = self.mi.Bitmap(image)
        bmp8 = bmp.convert(self.mi.Bitmap.PixelFormat.RGB,
                           self.mi.Struct.Type.UInt8,
                           srgb_gamma=True)
        arr8 = np.array(bmp8, copy=False)  # [H, W, 3], uint8
        
        img = Image.fromarray(arr8, mode="RGB")
        result_path = self.path_manager.get_result_path("rgb")
        img.save(result_path, format="PNG")


class RenderDepth:
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        self.mi = mi
        self.path_manager = PathManager

    def render(self):
        scene_path = self.path_manager.get_scene_path("depth")
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

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)
        
        # Guardar como archivo numpy
        result_path = self.path_manager.get_result_path("depth")
        np.save(result_path, grayscale_channel)


class RenderThermal:
    def __init__(self):
        import mitsuba as mi
        mi.set_variant('cuda_ad_spectral')
        self.mi = mi
        self.path_manager = PathManager

    def render(self):
        scene_path = self.path_manager.get_scene_path("thermal")
        scene = self.mi.load_file(scene_path)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        contribution_path = self.path_manager.get_result_path("contribution_blackbody_air")
        contribution_blackbody_air = np.load(contribution_path)

        image_array = image_array + contribution_blackbody_air

        # Guardar toda la información (todos los canales)
        result_path = self.path_manager.get_result_path("thermal")
        np.save(result_path, image_array)

    def render_blackbody_air(self):
        scene_path = self.path_manager.get_scene_path("blackbody_air")
        scene = self.mi.load_file(scene_path)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        result_path = self.path_manager.get_result_path("blackbody_air")
        np.save(result_path, image_array)

    def render_transmittance_blackbody_air(self):
        scene_path = self.path_manager.get_scene_path("transmittance_blackbody_air")
        scene = self.mi.load_file(scene_path)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        result_path = self.path_manager.get_result_path("transmittance_blackbody_air")
        np.save(result_path, image_array)

    def render_temperature_map(self):
        scene_path = self.path_manager.get_scene_path("temperature_map")
        scene = self.mi.load_file(scene_path)

        image = self.mi.render(scene)

        # Convertir la imagen a numpy array
        image_array = np.array(image)

        # Colapsar bandas: promedio a lo largo del último eje -> [y, x]
        if image_array.ndim == 3:
            image_array = image_array.mean(axis=-1)
        else:
            image_array = image_array.squeeze()

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        result_path = self.path_manager.get_result_path("temperature_map")
        np.save(result_path, image_array)

