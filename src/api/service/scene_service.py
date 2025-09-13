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
from src.mitsuba_core.object_utils import ObjectUtils
from src.mitsuba_core.scene_parser import SceneParser
from src.mitsuba_core.sensor_utils import create_specfilm_bands
import numpy as np
import random
import src.config as config

object_utils = ObjectUtils()

logger = logging.getLogger(__name__)


class SceneService:
    def __init__(self):
        self.logger = logger
        self.scene_parser = SceneParser()

    def load_scene(self, file: UploadFile):
        # Verificar si el archivo es un ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un archivo ZIP")

        # Eliminar todo el contenido del directorio
        if os.path.exists(config.OUTPUT_STATIC_DIR):
            try:
                shutil.rmtree(config.OUTPUT_STATIC_DIR)
                os.makedirs(config.OUTPUT_STATIC_DIR)
            except OSError as e:
                self.logger.warning(f"Error al limpiar directorio {config.OUTPUT_STATIC_DIR}: {e}")
                os.makedirs(config.OUTPUT_STATIC_DIR)

        # guardar archivo zip
        with open(config.SCENE_ZIP, "wb") as buffer:
            buffer.write(file.file.read())

        # descomprimir archivo zip
        with zipfile.ZipFile(config.SCENE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(config.OUTPUT_STATIC_DIR)

        # buscar y renombrar .xml
        xml_files = glob.glob(os.path.join(config.OUTPUT_STATIC_DIR, "*.xml"))
        if len(xml_files) != 1:
            raise HTTPException(status_code=400, detail="Debe haber exactamente un archivo XML en el ZIP")

        xml_file = xml_files[0]
        os.rename(xml_file, config.SCENE_DIR)

        config_scene = {}
        num_bands = 10
        T = 300

        # get material
        material = object_utils.get_material_signature("stone")
        indices = np.linspace(0, len(material) - 1, num_bands, dtype=int)
        metarial = material[indices]

        scene_dict = self.get_dict_scene(config.SCENE_DIR)

        # cambiar el spp de la scenea por defecto
        scene_dict['scene']['default'][0]['@value'] = 256

        # cambiar el width y height de la escena por defecto
        # resx -> width
        scene_dict['scene']['default'][1]['@value'] = 256

        # resy -> height
        scene_dict['scene']['default'][2]['@value'] = 256

        # guardar cambios en scene.xml
        self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_DIR)

        # guardar scene_dict como JSON
        json_output_path = os.path.join(config.OUTPUT_DIR, "scene.json")
        with open(json_output_path, 'w') as json_file:
            json.dump(scene_dict, json_file, indent=2)

        # revisar si se encuentran los archivos de los objetos y crear firmas
        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
            shapes = scene_dict["scene"]["shape"]
            if isinstance(shapes, list):
                for shape in shapes:
                    id = shape["string"]["@value"]
                    obj = {}
                    obj["emissivity"] = metarial.tolist()
                    obj["temperature"] = T
                    config_scene[id] = obj

                    dir_file = os.path.join(config.OUTPUT_STATIC_DIR, id)
                    if not os.path.exists(dir_file):
                        raise HTTPException(status_code=400, detail=f"No se encontró el archivo del objeto: {id}")

            elif isinstance(shapes, dict):
                id = shapes["string"]["@value"]
                obj = {}
                obj["id"] = id
                obj["emissivity"] = metarial.tolist()
                obj["temperature"] = T
                config_scene["objects"].append(obj)

                dir_file = os.path.join(config.OUTPUT_STATIC_DIR, id)

                if not os.path.exists(dir_file):
                    raise HTTPException(status_code=400, detail=f"No se encontró el archivo del objeto: {id}")

        # air values
        t_air = 280
        wavelengths = np.linspace(8000, 14000, num_bands)
        sigma_t_values = object_utils.get_attenuation_for_wavelengths(wavelengths)

        config_scene["air"] = {
            "temperature": t_air,
            "sigma_t": sigma_t_values.tolist()
        }

        config_scene["camera"] = {
            "spp": 256,
            "width": 256,
            "height": 256
        }

        config_scene["num_bands"] = num_bands
        config_scene["wavelengths"] = wavelengths.tolist()

        # guardar config_scene como JSON
        with open(config.CONFIG_SCENE, 'w') as f:
            json.dump(config_scene, f, indent=4)

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
        # Cargar configuración JSON desde el directorio de la escena
        config_path = os.path.join(os.path.dirname(scene_path), "config_scene.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=400, detail="No se encontró el archivo config_scene.json")

        with open(config_path, 'r') as f:
            config_scene = json.load(f)

        num_bands = config_scene["num_bands"]
        wavelengths = config_scene["wavelengths"]

        # Validar que num_bands coincida con la longitud de wavelengths
        if len(wavelengths) != num_bands:
            raise HTTPException(
                status_code=400, 
                detail=f"El número de bandas ({num_bands}) no coincide con las longitudes ({len(wavelengths)})"
            )

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

                # Agregar emisor diferente a cada shape
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        # para cada objeto
                        id = shape["string"]["@value"]
                        material = config_scene[id]["emissivity"]
                        temperature = config_scene[id]["temperature"]
                        radiance = object_utils.blackbody_radiance_nm(wavelengths, temperature)
                        emission = radiance * material
                        dict_emission = object_utils.create_spectral_emitter(wavelengths, emission)
                        shape["emitter"] = dict_emission

                        # Validar que la longitud del material coincida con el número de bandas
                        if len(material) != num_bands:
                            raise HTTPException(
                                status_code=400,
                                detail=f"La firma espectral ({id}) no coincide con el número de bandas ({num_bands})"
                            )
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape
                    id = shape["string"]["@value"]
                    material = config_scene[id]["emissivity"]
                    temperature = config_scene[id]["temperature"]
                    radiance = object_utils.blackbody_radiance_nm(wavelengths, temperature)
                    emission = radiance * material
                    dict_emission = object_utils.create_spectral_emitter(wavelengths, emission)
                    shape["emitter"] = dict_emission

                    if len(material) != num_bands:
                            raise HTTPException(
                                status_code=400,
                                detail=f"La firma espectral ({id}) no coincide con el número de bandas ({num_bands})"
                            )

            # Agregar medio homogéneo con coeficiente de extinción espectral
            # Crear valores de sigma_t para el medium (coeficiente de extinción)
            # Valores típicos para niebla en infrarrojo lejano
            sigma_t_values = config_scene["air"]["sigma_t"] # air values

            # Validar que sigma_t_values tenga la longitud correcta
            if len(sigma_t_values) != num_bands:
                raise HTTPException(
                    status_code=400,
                    detail=f"Los valores de sigma_t ({len(sigma_t_values)}) no coinciden con el número de bandas ({num_bands})"
                )
            
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


        
