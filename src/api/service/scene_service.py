import logging

from fastapi import UploadFile

from fastapi import HTTPException
from ...config import get_output_path
import os
import glob
import zipfile
import shutil
import json
from fastapi import UploadFile, HTTPException
from ...config import get_output_path
from ...mitsuba_core.object_utils import ObjectUtils
from ...mitsuba_core.scene_parser import SceneParser
from ...mitsuba_core.sensor_utils import create_specfilm_bands
import numpy as np
import random

object_utils = ObjectUtils()

logger = logging.getLogger(__name__)

out_dir = get_output_path("static")


class SceneService:
    def __init__(self):
        self.logger = logger
        self.scene_parser = SceneParser()

    def load_scene(self, file: UploadFile):
        # Verificar si el archivo es un ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un archivo ZIP")

        # Eliminar todo el contenido del directorio
        if os.path.exists(out_dir):
            for item in os.listdir(out_dir):
                item_path = os.path.join(out_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)    
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except OSError as e:
                    self.logger.warning(f"Error al eliminar {item_path}: {e}")
        
        # guardar archivo zip
        with open(f"{out_dir}/scene.zip", "wb") as buffer:
            buffer.write(file.file.read())

        # descomprimir archivo zip
        with zipfile.ZipFile(f"{out_dir}/scene.zip", 'r') as zip_ref:
            zip_ref.extractall(out_dir)

        # buscar y renombrar .xml
        xml_files = glob.glob(os.path.join(out_dir, "*.xml"))
        if len(xml_files) != 1:
            raise HTTPException(status_code=400, detail="Debe haber exactamente un archivo XML en el ZIP")

        xml_file = xml_files[0]
        new_xml_path = os.path.join(out_dir, "scene.xml")
        os.rename(xml_file, new_xml_path)

        return new_xml_path

    def has_loaded_scene(self, scene_dir: str) -> bool:
        """
        Verifica si se ha cargado una escena en el directorio especificado.
        
        Args:
            scene_dir: Ruta del directorio de la escena.
        
        Returns:
            True si se ha cargado una escena, False en caso contrario.
        """
        return os.path.exists(scene_dir) and os.path.isfile(scene_dir)

    def prepare_thermal_scene(self, scene_path: str):
        """
        Crea una escena térmica básica copiando scene.xml a scene_thermal.xml y cambiando el tipo de integrador a 'specfilm'.
        """
        num_bands = 10 
        w_min = 8000
        w_max = 14000
        wavelengths = np.linspace(w_min, w_max, num_bands)

        if os.path.exists(scene_path):
            thermal_xml = os.path.join(os.path.dirname(scene_path), "scene_thermal.xml")
            shutil.copyfile(scene_path, thermal_xml)
            scene_dict = self.get_dict_scene(thermal_xml)

            # Cambiar el tipo de film
            if scene_dict and "scene" in scene_dict and "sensor" in scene_dict["scene"]:
                if "film" in scene_dict["scene"]["sensor"]:
                    scene_dict["scene"]["sensor"]["film"]["@type"] = "specfilm"
                    # Agregar bandas espectrales al film
                    bands = create_specfilm_bands(wavelengths)
                    # Usar la estructura que retorna create_specfilm_bands directamente
                    scene_dict["scene"]["sensor"]["film"]["spectrum"] = bands
                
                # Agregar referencia al medium en el sensor para que vea a través del gas
                scene_dict["scene"]["sensor"]["ref"] = {
                    "@name": "medium",
                    "@id": "fog"
                }
            
            # Cambiar el integrador a volpathmis con estructura correcta
            if scene_dict and "scene" in scene_dict:
                scene_dict["scene"]["integrator"] = {
                    "@type": "volpathmis",
                    "integer": {
                        "@name": "max_depth",
                        "@value": "16"
                    }
                }
            # Eliminar los materiales bsdf
            if scene_dict and "scene" in scene_dict and "bsdf" in scene_dict["scene"]:
                del scene_dict["scene"]["bsdf"]
            # Eliminar el emisor emitter
            if scene_dict and "scene" in scene_dict and "emitter" in scene_dict["scene"]:
                del scene_dict["scene"]["emitter"]
            # Eliminar referencias 'ref' en cada shape
            if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]
                if isinstance(shapes, list):
                    for shape in shapes:
                        if "ref" in shape:
                            del shape["ref"]
                elif isinstance(shapes, dict):
                    if "ref" in shapes:
                        del shapes["ref"]
                # get material
                material = object_utils.get_material_signature("stone")
                indices = np.linspace(0, len(material) - 1, num_bands, dtype=int)
                metarial = material[indices]

                # Agregar emisor diferente a cada shape
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        # Temperatura aleatoria diferente para cada objeto
                        temperature = random.randint(200, 400)
                        radiance = object_utils.blackbody_radiance_nm(wavelengths, temperature)
                        emission = radiance * metarial
                        dict_emission = object_utils.create_spectral_emitter(wavelengths, emission)
                        shape["emitter"] = dict_emission
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape
                    temperature = random.randint(200, 400)
                    radiance = object_utils.blackbody_radiance_nm(wavelengths, temperature)
                    emission = radiance * metarial
                    dict_emission = object_utils.create_spectral_emitter(wavelengths, emission)
                    shapes["emitter"] = dict_emission

            # Agregar medio homogéneo con coeficiente de extinción espectral
            # Crear valores de sigma_t para el medium (coeficiente de extinción)
            # Valores típicos para niebla en infrarrojo lejano
            sigma_t_values = object_utils.get_attenuation_for_wavelengths(wavelengths) # air values
            
            # Crear el medium usando la función de object_utils
            medium_dict = object_utils.create_homogeneous_medium(
                wavelengths=wavelengths,
                sigma_t=sigma_t_values,
                medium_id="fog",
                g_value=0.95  # Valor para infrarrojo lejano
            )
            
            # Agregar el medium a la escena
            scene_dict["scene"]["medium"] = medium_dict


            # Guardar la escena modificada como XML
            if scene_dict:
                self.scene_parser.save_dict_as_xml(scene_dict, thermal_xml)

    def prepare_depth_scene(self, scene_path: str):
        """
        Prepara la escena para la renderización en profundidad.
        """
        if os.path.exists(scene_path):
            depth_xml = os.path.join(os.path.dirname(scene_path), "scene_depth.xml")
            shutil.copyfile(scene_path, depth_xml)
            scene_dict = self.get_dict_scene(depth_xml)
            if scene_dict and "scene" in scene_dict:
                # Modificar la escena según sea necesario para la renderización en profundidad
                scene_dict["scene"]["integrator"] = {
                    "@type": "depth",
                }
            
            if scene_dict:
                self.scene_parser.save_dict_as_xml(scene_dict, depth_xml)

    def get_dict_scene(self, scene_xml_path: str):
        """
        Carga cualquier archivo de escena XML y lo retorna como dict.
        """
        if not os.path.exists(scene_xml_path):
            return None
        scene_dict = self.scene_parser.xml_to_dict(scene_xml_path)
        return scene_dict


        
