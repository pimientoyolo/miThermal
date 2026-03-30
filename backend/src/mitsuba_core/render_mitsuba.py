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

        # 1. Normalización Espectral: Mitsuba integra sobre el ancho de banda
        # Necesitamos dividir por el ancho de banda (delta) y ajustar por el factor pi
        # para obtener Radiancia Espectral (W/m²/sr/m)
        from src.config import get_config_scene_dict
        config_scene = get_config_scene_dict()
        wavelengths = config_scene.get("wavelengths", [])
        if len(wavelengths) > 1:
            delta = float(wavelengths[1] - wavelengths[0])
            # Normalizar: (Valor / delta) * pi
            # El factor pi es necesario para convertir la proyección Lambertiana a Radiancia
            #image_array = (image_array * np.pi) / delta
            image_array = (image_array) / delta

        # 2. Recortar bandas de seguridad (primera y última)
        if image_array.ndim == 3 and image_array.shape[-1] > 2:
            image_array = image_array[:, :, 1:-1]

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # GUARDAR CUBE TÉRMICO PURO (antes de sumar aire) para diagnóstico
        result_raw_path = self.path_manager.get_result_path("thermal_raw")
        np.save(result_raw_path, image_array)

        contribution_path = self.path_manager.get_result_path("contribution_blackbody_air")
        contribution_blackbody_air = np.load(contribution_path)

        # Verificar compatibilidad de formas para evitar errores de broadcasting
        if image_array.shape != contribution_blackbody_air.shape:
            # Caso común: image_array es (H, W, 3) y contribución es (H, W, Bands)
            if image_array.shape[-1] == 3 and contribution_blackbody_air.shape[-1] != 3:
                # Si Mitsuba devolvió RGB pero tenemos bandas, colapsamos bandas de contribución a promedio
                # para que el usuario al menos vea un resultado coherente en intensidad
                avg_contrib = np.mean(contribution_blackbody_air, axis=-1, keepdims=True)
                # Opcional: repetir 3 veces para sumar a RGB
                avg_contrib_rgb = np.repeat(avg_contrib, 3, axis=-1)
                image_array = image_array + avg_contrib_rgb
            else:
                # Otros casos: intentar sumar directamente (fallará si no son compatibles)
                try:
                    image_array = image_array + contribution_blackbody_air
                except ValueError as e:
                    logging.error(f"Error de broadcasting: image={image_array.shape}, contrib={contribution_blackbody_air.shape}")
                    # Mantener solo el render de Mitsuba si falla
                    pass
        else:
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

        # Normalización
        from src.config import get_config_scene_dict
        config_scene = get_config_scene_dict()
        wavelengths = config_scene.get("wavelengths", [])
        if len(wavelengths) > 1:
            delta = float(wavelengths[1] - wavelengths[0])
            #image_array = (image_array * np.pi) / delta
            image_array = (image_array) / delta

        # Recortar bandas de seguridad
        if image_array.ndim == 3 and image_array.shape[-1] > 2:
            image_array = image_array[:, :, 1:-1]

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

        # Normalización
        from src.config import get_config_scene_dict
        config_scene = get_config_scene_dict()
        wavelengths = config_scene.get("wavelengths", [])
        if len(wavelengths) > 1:
            delta = float(wavelengths[1] - wavelengths[0])
            #image_array = (image_array * np.pi) / delta
            image_array = (image_array) / delta

        # Recortar bandas de seguridad
        if image_array.ndim == 3 and image_array.shape[-1] > 2:
            image_array = image_array[:, :, 1:-1]

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

        # Para el mapa de temperatura, la banda es de 1nm, por lo que dividimos entre 1.0 (delta)
        # No recortamos bandas de seguridad porque es una escena monocromática custom.
        # No multiplicamos por pi porque queremos el valor crudo de temperatura.
        image_array = image_array / 1.0

        # Colapsar bandas: promedio a lo largo del último eje -> [y, x]
        # Al ser un mapa de temperatura, todas las bandas deberían tener el mismo valor
        if image_array.ndim == 3:
            image_array = image_array.mean(axis=-1)
        else:
            image_array = image_array.squeeze()

        # Crear carpeta si no existe
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)

        # Guardar toda la información (todos los canales)
        result_path = self.path_manager.get_result_path("temperature_map")
        np.save(result_path, image_array)

