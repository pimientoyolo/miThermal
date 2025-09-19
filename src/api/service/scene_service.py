import logging

from fastapi import UploadFile

from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.api.dto.cameraDTO import CameraDTO
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
        T = 300

        scene_dict = self.get_dict_scene(config.SCENE_DIR)

        # cambiar el spp de la scenea por defecto
        scene_dict['scene']['default'][0]['@value'] = 256

        # cambiar el width y height de la escena por defecto
        # resx -> width
        scene_dict['scene']['default'][1]['@value'] = 256

        # resy -> height
        scene_dict['scene']['default'][2]['@value'] = 256

        # cambiar el tipo de emiter a uno de uniforme luz ambiente
        if "emitter" in scene_dict["scene"]:
            scene_dict["scene"]["emitter"] = {
                "@type": "constant",
                "rgb": {
                    "@name": "radiance",
                    "@value": "1.0"
                }
            }


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
                    object_utils.save_default_emissivity(id)
                    obj = {}
                    obj["temperature"] = T
                    config_scene[id] = obj

                    dir_file = os.path.join(config.OUTPUT_STATIC_DIR, id)
                    if not os.path.exists(dir_file):
                        raise HTTPException(status_code=400, detail=f"No se encontró el archivo del objeto: {id}")

            elif isinstance(shapes, dict):
                id = shapes["string"]["@value"]
                object_utils.save_default_emissivity(id)
                obj = {}
                obj["id"] = id
                obj["temperature"] = T
                config_scene["objects"].append(obj)

                dir_file = os.path.join(config.OUTPUT_STATIC_DIR, id)

                if not os.path.exists(dir_file):
                    raise HTTPException(status_code=400, detail=f"No se encontró el archivo del objeto: {id}")

        # air values
        t_air = 280

        wavelengths = np.linspace(10000, 12000, 10, endpoint=True, dtype=int)

        # Obtener longitudes de onda y atenuación desde archivo de referencia
        object_utils.get_attenuation()
        num_bands = len(wavelengths)

        config_scene["air"] = {
            "temperature": t_air
        }
        rx = ry = rz = 0.0
        tx = ty = tz = 0.0

        # obtener sensor y transform
        sensor = scene_dict["scene"].get("sensor")
        transform = sensor.get("transform")

        # obtner las rotaciones
        rotate = transform.get("rotate")

        # Asegurar que exista transform
        if transform is None:
            sensor["transform"] = {}
            transform = sensor["transform"]

        rotate = transform.get("rotate")

        # Leer valores actuales si existen
        if isinstance(rotate, list):
            for r in rotate:
                a = float(r.get("@angle", 0.0))
                if r.get("@x") == "1":
                    rx = a
                elif r.get("@y") == "1":
                    ry = a
                elif r.get("@z") == "1":
                    rz = a

        elif isinstance(rotate, dict):
            a = float(rotate.get("@angle", 0.0))
            if rotate.get("@x") == "1":
                rx = a
            elif rotate.get("@y") == "1":
                ry = a
            elif rotate.get("@z") == "1":
                rz = a

        # Normalizar/crear rotate como lista de 3 entradas
        transform["rotate"] = [
            {"@x": "1", "@angle": str(rx)},
            {"@y": "1", "@angle": str(ry)},
            {"@z": "1", "@angle": str(rz)}
        ]

        # obtener las traslaciones
        translate = transform.get("translate")

        value = translate.get("@value", "0 0 0")
        coords = value.split()
        tx = float(coords[0])
        ty = float(coords[1])
        tz = float(coords[2])

        # obtener el fov
        floats = sensor.get("float")

        fov = 45.0

        if isinstance(floats, list):
            for f in floats:
                if f.get("@name") == "fov":
                    fov = float(f.get("@value"))
        elif isinstance(floats, dict):
            if floats.get("@name") == "fov":
                fov = float(floats.get("@value"))

        angles = mitsuba_cam_to_blender_xyz(rx, ry, rz)

        config_scene["camera"] = {
            "spp": 256,
            "width": 256,
            "height": 256,
            "rotate_x": angles[0], # ajuste para mitsuba
            "rotate_y": angles[1], # ajuste para mitsuba
            "rotate_z": angles[2], # ajuste para mitsuba
            "translate_x": tx, # x mitsuba igual al x blender
            "translate_y": tz, # y mitsuba igual al z blender
            "translate_z": -ty, # eje z igual al eje -y de blender
            "fov": fov
        }
        
        config_scene["num_bands"] = num_bands
        config_scene["wavelengths"] = wavelengths.tolist()

        # guardar cambios en scene.xml
        self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_DIR)

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

    def prepare_thermal_scene(self):
        """
        Crea una escena térmica básica copiando scene.xml a scene_thermal.xml y cambiando el tipo de integrador a 'specfilm'.
        """
        # Cargar configuración JSON desde el directorio de la escena
        config_scene = config.get_config_scene_dict()

        num_bands = config_scene["num_bands"]
        wavelengths = config_scene["wavelengths"]

        # Validar que num_bands coincida con la longitud de wavelengths
        if len(wavelengths) != num_bands:
            raise HTTPException(
                status_code=400, 
                detail=f"El número de bandas ({num_bands}) no coincide con las longitudes ({len(wavelengths)})"
            )

        if os.path.exists(config.SCENE_DIR):
            thermal_xml = config.SCENE_THERMAL_DIR
            shutil.copyfile(config.SCENE_DIR, thermal_xml)
            scene_dict = self.get_dict_scene(thermal_xml)

            # Cambiar el tipo de film y configurar sampler
            if scene_dict and "scene" in scene_dict and "sensor" in scene_dict["scene"]:
                if "film" in scene_dict["scene"]["sensor"]:
                    scene_dict["scene"]["sensor"]["film"]["@type"] = "specfilm"
                    # Agregar bandas espectrales al film
                    bands = create_specfilm_bands(wavelengths)
                    # Usar la estructura que retorna create_specfilm_bands directamente
                    scene_dict["scene"]["sensor"]["film"]["spectrum"] = bands
                    
                    # Agregar filtro gaussiano para mejor suavizado de la imagen
                    scene_dict["scene"]["sensor"]["film"]["rfilter"] = {
                        "@type": "gaussian",
                        "float": {
                            "@name": "stddev",
                            "@value": "0.5"
                        }
                    }
                
                # Cambiar el sampler a multijitter para mejor calidad de muestreo
                if "sampler" in scene_dict["scene"]["sensor"]:
                    scene_dict["scene"]["sensor"]["sampler"]["@type"] = "multijitter"
                    
                
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
                        wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(id)
                        temperature = config_scene[id]["temperature"]
                        radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                        emission = radiance * emissivity
                        reflectance = 1 - np.array(emissivity)
                        dict_reflectance = object_utils.create_reflectance_material(wavelengths_obj, reflectance)
                        dict_emission = object_utils.create_spectral_emitter(wavelengths_obj, emission)
                        shape["emitter"] = dict_emission
                        shape['bsdf'] = dict_reflectance
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape
                    id = shapes["string"]["@value"]
                    wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(id)
                    temperature = config_scene[id]["temperature"]
                    radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                    emission = radiance * emissivity
                    reflectance = 1 - np.array(emissivity)
                    dict_reflectance = object_utils.create_reflectance_material(wavelengths_obj, reflectance)
                    dict_emission = object_utils.create_spectral_emitter(wavelengths_obj, emission)
                    shapes["emitter"] = dict_emission
                    shapes['bsdf'] = dict_reflectance


            # Agregar medio homogéneo con coeficiente de extinción espectral
            # Crear valores de sigma_t para el medium (coeficiente de extinción)
            # Valores típicos para niebla en infrarrojo lejano
            t_air = config_scene["air"]["temperature"]
            wavelengths_air, sigma_t_values = object_utils.read_air_attenuation_file()

            emission_air = object_utils.blackbody_radiance_nm(wavelengths_air, t_air)

            emitter_air_dict = object_utils.create_spectral_emitter(wavelengths_air, emission_air, "constant")

            # Agregar emisor de aire a la escena
            scene_dict["scene"]["emitter"] = emitter_air_dict
            
            # Crear el medium usando la función de object_utils
            medium_dict = object_utils.create_homogeneous_medium(
                wavelengths=wavelengths_air,
                sigma_t=sigma_t_values,
                medium_id="fog",
                g_value=0.95  # Valor para infrarrojo lejano
            )

            
            # Agregar el medium a la escena
            scene_dict["scene"]["medium"] = medium_dict


            # Guardar la escena modificada como XML
            if scene_dict:
                self.scene_parser.save_dict_as_xml(scene_dict, thermal_xml)

    def prepare_depth_scene(self):
        """
        Prepara la escena para la renderización en profundidad.
        """
        depth_xml = config.SCENE_DEPTH_DIR
        shutil.copyfile(config.SCENE_DIR, depth_xml)
        scene_dict = self.get_dict_scene(depth_xml)
        if scene_dict and "scene" in scene_dict:
            # Modificar la escena según sea necesario para la renderización en profundidad
            scene_dict["scene"]["integrator"] = {
                "@type": "depth",
            }
        
        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, depth_xml)

    def prepare_blackbody_air_scene(self):
        """
        Prepara la escena para renderización de cuerpo negro del aire.
        """
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(config.SCENE_THERMAL_DIR):
            self.prepare_thermal_scene()

        # Crear la escena de blackbody air copiando la escena térmica
        blackbody_air_xml = config.SCENE_BLACKBODY_AIR
        shutil.copyfile(config.SCENE_THERMAL_DIR, blackbody_air_xml)
        scene_dict = self.get_dict_scene(blackbody_air_xml)

        # Eliminar todos los objetos de la escena (shapes)
        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
            del scene_dict["scene"]["shape"]

        # Eliminar el medio atenuante
        if scene_dict and "scene" in scene_dict and "medium" in scene_dict["scene"]:
            del scene_dict["scene"]["medium"]

        if scene_dict and "scene" in scene_dict and "sensor" in scene_dict["scene"]:
            del scene_dict["scene"]["sensor"]["ref"]

        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, blackbody_air_xml)

    def prepare_transmittance_blackbody_air_scene(self):
        """
        Prepara la escena para renderización de transmitancia de cuerpo negro del aire.
        """
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(config.SCENE_THERMAL_DIR):
            self.prepare_thermal_scene()

        # Crear la escena de transmitancia blackbody air copiando la escena térmica
        transmittance_blackbody_air_xml = config.SCENE_TRANSMITTANCE_BLACKBODY_AIR
        shutil.copyfile(config.SCENE_THERMAL_DIR, transmittance_blackbody_air_xml)
        scene_dict = self.get_dict_scene(transmittance_blackbody_air_xml)

        config_scene = config.get_config_scene_dict()

        t_air = config_scene["air"]["temperature"]
        wavelengths_air, _ = object_utils.read_air_attenuation_file()

        emission_air = object_utils.blackbody_radiance_nm(wavelengths_air, t_air)

        emitter_air_dict = object_utils.create_spectral_emitter(wavelengths_air, emission_air)

        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]

                # Agregar emisor diferente a cada shape
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        
                        shape["emitter"] = emitter_air_dict
                        del shape["bsdf"]
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape

                    shapes["emitter"] = emitter_air_dict
                    del shapes['bsdf']

        # eliminar emisor de aire
        if scene_dict and "scene" in scene_dict and "emitter" in scene_dict["scene"]:
            del scene_dict["scene"]["emitter"]


        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, transmittance_blackbody_air_xml)

    def prepare_temperature_map(self):
        """
        Prepara la escena para renderización de la temperatura de los objetos.
        """
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(config.SCENE_THERMAL_DIR):
            self.prepare_thermal_scene()

        # Crear la escena de transmitancia blackbody air copiando la escena térmica
        temperature_map_xml = config.SCENE_TEMPERATURE_MAP
        shutil.copyfile(config.SCENE_THERMAL_DIR, temperature_map_xml)
        scene_dict = self.get_dict_scene(temperature_map_xml)

        config_scene = config.get_config_scene_dict()

        wavelengths = config_scene["wavelengths"]


        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]

                # Agregar emisor diferente a cada shape
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        id = shape["string"]["@value"]
                        temperature = config_scene[id]["temperature"]
                        emission = [temperature for _ in wavelengths]
                        shape["emitter"] = object_utils.create_spectral_emitter(wavelengths, emission)
                        del shape["bsdf"]
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape
                    id = shapes["string"]["@value"]
                    temperature = config_scene[id]["temperature"]
                    emission = [temperature for _ in wavelengths]
                    shapes["emitter"] = object_utils.create_spectral_emitter(wavelengths, emission)
                    del shapes['bsdf']

        # Cambiar el integrador a path para reducir ruido
        if scene_dict and "scene" in scene_dict:
            scene_dict["scene"]["integrator"] = {
                "@type": "path",
                "integer": {
                    "@name": "max_depth",
                    "@value": "16"
                }
            }

        # Eliminar el medio atenuante
        if scene_dict and "scene" in scene_dict and "medium" in scene_dict["scene"]:
            del scene_dict["scene"]["medium"]

        if scene_dict and "scene" in scene_dict and "sensor" in scene_dict["scene"]:
            del scene_dict["scene"]["sensor"]["ref"]

        # eliminar emisor de aire
        if scene_dict and "scene" in scene_dict and "emitter" in scene_dict["scene"]:
            del scene_dict["scene"]["emitter"]

        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, temperature_map_xml)

    def get_dict_scene(self, scene_xml_path: str):
        """
        Carga cualquier archivo de escena XML y lo retorna como dict.
        """
        if not os.path.exists(scene_xml_path):
            return None
        scene_dict = self.scene_parser.xml_to_dict(scene_xml_path)
        return scene_dict
    
    def update_thermal_scene_obj(self, object_id: str):
        """
        Actualiza un objeto específico en la escena térmica con sus nuevas propiedades.
        """
        # Cargar configuración JSON desde el directorio de la escena
        config_scene = config.get_config_scene_dict()

        scene_dict = self.get_dict_scene(config.SCENE_THERMAL_DIR)
        
        # Verificar que el objeto existe en la configuración
        if object_id not in config_scene:
            raise HTTPException(
                status_code=404,
                detail=f"El objeto {object_id} no se encuentra en la configuración de la escena"
            )
        
        # Obtener las propiedades del objeto específico
        wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(object_id)
        temperature = config_scene[object_id]["temperature"]
        
        
        object_found = False
        
        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
            shapes = scene_dict["scene"]["shape"]
            
            # Buscar y actualizar solo el objeto específico
            if isinstance(shapes, list):
                for i, shape in enumerate(shapes):
                    shape_id = shape["string"]["@value"]
                    if shape_id == object_id:
                        # Actualizar solo este objeto
                        radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                        emission = radiance * emissivity
                        reflectance = 1 - np.array(emissivity)
                        dict_reflectance = object_utils.create_reflectance_material(wavelengths_obj, reflectance)
                        dict_emission = object_utils.create_spectral_emitter(wavelengths_obj, emission)
                        shape["emitter"] = dict_emission
                        shape['bsdf'] = dict_reflectance
                        object_found = True
                        break

            elif isinstance(shapes, dict):
                # Para un solo shape
                shape_id = shapes["string"]["@value"]
                if shape_id == object_id:
                    radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                    emission = radiance * emissivity
                    reflectance = 1 - np.array(emissivity)
                    dict_reflectance = object_utils.create_reflectance_material(wavelengths_obj, reflectance)
                    dict_emission = object_utils.create_spectral_emitter(wavelengths_obj, emission)
                    shapes["emitter"] = dict_emission
                    shapes['bsdf'] = dict_reflectance
                    object_found = True
        
        if not object_found:
            raise HTTPException(
                status_code=404,
                detail=f"El objeto {object_id} no se encuentra en la escena térmica"
            )
        
        # Guardar la escena térmica actualizada
        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_THERMAL_DIR)

    def update_thermal_scene_air(self):
        """
        Actualiza las propiedades del aire en la escena térmica con sus nuevas propiedades.
        Refresca el espectro del film (sensor espectral) según las nuevas longitudes de onda.
        """
        config_scene = config.get_config_scene_dict()

        scene_dict = self.get_dict_scene(config.SCENE_THERMAL_DIR)

        if not scene_dict or "scene" not in scene_dict:
            raise HTTPException(status_code=404, detail="Escena térmica no encontrada para actualizar aire")

        # === Actualizar film spectrum (extraído a helper) ===
        self.update_film_spectrum()

        # Datos de aire
        t_air = config_scene["air"]["temperature"]
        wavelengths, sigma_t_values = object_utils.read_air_attenuation_file()

        # Recalcular emisión del aire
        emission_air = object_utils.blackbody_radiance_nm(wavelengths, t_air)
        emitter_air_dict = object_utils.create_spectral_emitter(wavelengths, emission_air, "constant")
        scene_dict["scene"]["emitter"] = emitter_air_dict

        # Medium actualizado
        medium_dict = object_utils.create_homogeneous_medium(
            wavelengths=wavelengths,
            sigma_t=sigma_t_values,
            medium_id="fog",
            g_value=0.95
        )
        scene_dict["scene"]["medium"] = medium_dict

        # Guardar la escena modificada como XML
        self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_THERMAL_DIR)

    def update_film_spectrum(self) -> bool:
        """
        Actualiza el film del sensor para que use bandas espectrales acorde a 'wavelengths'.
        - Asegura el tipo 'specfilm' si ensure_specfilm=True

        Retorna True si se actualizó, False si no se encontró sensor/film.
        """
        config_scene = config.get_config_scene_dict()
        wavelengths = config_scene["wavelengths"]

        scene_dict = self.get_dict_scene(config.SCENE_THERMAL_DIR)

        if not scene_dict or "scene" not in scene_dict:
            return False
        sensor = scene_dict["scene"].get("sensor")
        if not sensor or "film" not in sensor:
            return False

        film = sensor["film"]
        film["@type"] = "specfilm"
        film["spectrum"] = create_specfilm_bands(wavelengths)

        self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_THERMAL_DIR)

        return True
    
    def update_scene_camera_rgb(self):
        scene_dict = self.get_dict_scene(config.SCENE_DIR)

        if not scene_dict or "scene" not in scene_dict or "default" not in scene_dict["scene"]:
            raise HTTPException(status_code=400, detail="No se encontraron parámetros por defecto en la escena")

        defaults = scene_dict["scene"]["default"]

        cam_cfg = config.get_config_scene_dict().get("camera", {})
        spp = cam_cfg.get("spp", 256)
        width = cam_cfg.get("width", 256)
        height = cam_cfg.get("height", 256)
        rx = float(cam_cfg.get("rotate_x", 0.0))
        ry = float(cam_cfg.get("rotate_y", 0.0))
        rz = float(cam_cfg.get("rotate_z", 0.0))
        tx = float(cam_cfg.get("translate_x", 0.0))
        ty = float(cam_cfg.get("translate_y", 0.0))
        tz = float(cam_cfg.get("translate_z", 0.0))
        fov = float(cam_cfg.get("fov", 45.0))

        defaults[0]["@value"] = spp      # spp
        defaults[1]["@value"] = width    # resx
        defaults[2]["@value"] = height   # resy

        scene = scene_dict["scene"]
        sensor = scene.get("sensor")
        transform = sensor.get("transform")

        angles = blender_cam_to_mitsuba_xyz(rx, ry, rz)

        # set rotate
        transform["rotate"] = [
            {
                "@x": "1",
                "@angle": str(angles[0])  # ajuste para mitsuba
            },
            {
                "@y": "1",
                "@angle": str(angles[1])  # ajuste para mitsuba
            },
            {
                "@z": "1",
                "@angle": str(angles[2])  # ajuste para mitsuba
            }
        ]

        # set translate
        transform["translate"] = {
            "@value": f"{tx} {tz} {-ty}"  # y mitsuba = z blender, z mitsuba = -y blender
        }

        # set fov
        floats = sensor.get("float")
        if isinstance(floats, list):
            fov_found = False
            for f in floats:
                if f.get("@name") == "fov":
                    f["@value"] = str(fov)
                    fov_found = True
                    break
            if not fov_found:
                floats.append({
                    "@name": "fov",
                    "@value": str(fov)
                })

        elif isinstance(floats, dict):
            if floats.get("@name") == "fov":
                floats["@value"] = str(fov)
            else:
                sensor["float"] = [
                    floats,
                    {
                        "@name": "fov",
                        "@value": str(fov)
                    }
                ]

        # Guardar la escena modificada como XML
        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_DIR)

    def update_scene_camera_thermal(self):
        scene_dict = self.get_dict_scene(config.SCENE_THERMAL_DIR)

        if not scene_dict or "scene" not in scene_dict or "default" not in scene_dict["scene"]:
            raise HTTPException(status_code=400, detail="No se encontraron parámetros por defecto en la escena térmica")

        defaults = scene_dict["scene"]["default"]

        cam_cfg = config.get_config_scene_dict().get("camera", {})
        spp = cam_cfg.get("spp", 256)
        width = cam_cfg.get("width", 256)
        height = cam_cfg.get("height", 256)
        rx = float(cam_cfg.get("rotate_x", 0.0))
        ry = float(cam_cfg.get("rotate_y", 0.0))
        rz = float(cam_cfg.get("rotate_z", 0.0))
        tx = float(cam_cfg.get("translate_x", 0.0))
        ty = float(cam_cfg.get("translate_y", 0.0))
        tz = float(cam_cfg.get("translate_z", 0.0))
        fov = float(cam_cfg.get("fov", 45.0))

        defaults[0]["@value"] = spp      # spp
        defaults[1]["@value"] = width    # resx
        defaults[2]["@value"] = height   # resy

        scene = scene_dict["scene"]
        sensor = scene.get("sensor")
        transform = sensor.get("transform")

        angles = blender_cam_to_mitsuba_xyz(rx, ry, rz)

        # set rotate
        transform["rotate"] = [
            {
                "@x": "1",
                "@angle": str(angles[0])  # ajuste para mitsuba
            },
            {
                "@y": "1",
                "@angle": str(angles[1])  # ajuste para mitsuba
            },
            {
                "@z": "1",
                "@angle": str(angles[2])  # ajuste para mitsuba
            }
        ]

        # set translate
        transform["translate"] = {
            "@value": f"{tx} {tz} {-ty}"  # y mitsuba = z blender, z mitsuba = -y blender
        }

        # set fov
        floats = sensor.get("float")
        if isinstance(floats, list):
            fov_found = False
            for f in floats:
                if f.get("@name") == "fov":
                    f["@value"] = str(fov)
                    fov_found = True
                    break
            if not fov_found:
                floats.append({
                    "@name": "fov",
                    "@value": str(fov)
                })

        elif isinstance(floats, dict):
            if floats.get("@name") == "fov":
                floats["@value"] = str(fov)
            else:
                sensor["float"] = [
                    floats,
                    {
                        "@name": "fov",
                        "@value": str(fov)
                    }
                ]

        # Guardar la escena modificada como XML
        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, config.SCENE_THERMAL_DIR)

    def get_scene_mi_thermal(self) -> FileResponse:
        
        path = str(config.OUTPUT_STATIC_DIR)
        zip_filename = config.MITHERMAL_SCENE_FILE 

        # eliminar el zip si ya existe
        if os.path.exists(zip_filename):
            os.remove(zip_filename)

        # crear el zip con zipfile
        with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_STORED, compresslevel=0) as zf:
            for root, _, files in os.walk(path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, path) 
                    zf.write(abs_path, rel_path)

        # devolverlo como respuesta
        return FileResponse(zip_filename, media_type="application/zip", filename="miThermal.zip")
    
    def upload_mi_thermal_scene(self, file: UploadFile) -> None:
        path = str(config.OUTPUT_STATIC_DIR)
        zip_path = config.MITHERMAL_SCENE_FILE

        # Vaciar el directorio 'path' antes de extraer el ZIP
        if os.path.exists(path):
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                try:
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
                except OSError as e:
                    self.logger.warning(f"Error al eliminar {entry_path}: {e}")
        else:
            os.makedirs(path, exist_ok=True)
        
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.OUTPUT_STATIC_DIR)
        
        os.remove(zip_path)
    
    def get_suggested_mi_thermal_scene(self) -> list[str]:
        
        path = config.DEFAULT_SCENES_DIR

        if not os.path.isdir(path):
            return []
        files = [os.path.basename(p) for p in glob.glob(os.path.join(path, "*.zip")) if os.path.isfile(p)]
        files.sort(key=str.lower)
        return files
    
    def set_default_scene(self, scene_name: str) -> None:
        """
        Copia un archivo ZIP de escena por defecto al directorio de salida y lo descomprime.
        """
        default_scene_path = os.path.join(config.DEFAULT_SCENES_DIR, scene_name)
        if not os.path.isfile(default_scene_path):
            raise HTTPException(status_code=404, detail=f"No se encontró la escena por defecto: {scene_name}")
        
        path = str(config.OUTPUT_STATIC_DIR)
        zip_filename = config.MITHERMAL_SCENE_FILE 

        # eliminar el zip si ya existe
        if os.path.exists(zip_filename):
            os.remove(zip_filename)

        # Vaciar el directorio 'path' antes de extraer el ZIP
        if os.path.exists(path):
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                try:
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
                except OSError as e:
                    self.logger.warning(f"Error al eliminar {entry_path}: {e}")
        else:
            os.makedirs(path, exist_ok=True)

        # copiar el archivo zip
        shutil.copyfile(default_scene_path, zip_filename)

        # extraer el zip
        with zipfile.ZipFile(zip_filename, "r") as zf:
            zf.extractall(path)


# --- Rotaciones básicas ---
def Rx(d): 
    r=np.deg2rad(d); c,s=np.cos(r),np.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)

def Ry(d):
    r=np.deg2rad(d); c,s=np.cos(r),np.sin(r)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)

def Rz(d):
    r=np.deg2rad(d); c,s=np.cos(r),np.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)

# --- Euler intrínseco XYZ con vectores-columna: R = Rz(z) @ Ry(y) @ Rx(x) ---
def euler_to_R_intrinsic_xyz(x,y,z):
    return Rz(z) @ Ry(y) @ Rx(x)

def clamp(v, lo=-1.0, hi=1.0): 
    return max(lo, min(hi, v))

def R_to_euler_intrinsic_xyz(R):
    """
    Descompone R = Rz(z) @ Ry(y) @ Rx(x) -> (x,y,z) en grados, normalizado a [-180,180).
    """
    y = np.arcsin(clamp(R[2,0]))
    cy = np.cos(y); eps=1e-8
    if abs(cy) > eps:
        x = np.arctan2(-R[2,1], R[2,2])
        z = np.arctan2(-R[1,0], R[0,0])
    else:
        # gimbal lock (y ≈ ±90°)
        z = 0.0
        x = np.arctan2(R[0,1], R[0,2])
    deg = np.rad2deg([x,y,z])
    return tuple(((deg+180.0)%360.0 - 180.0).tolist())

def wrap_deg(a): 
    return ((a + 180.0) % 360.0) - 180.0

# --- Cambio de base Blender→Mitsuba (tu hallazgo): X_M=X_B; Y_M=Z_B; Z_M=-Y_B ---
T = np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
Tinv = T.T  # inversa de rotación

def _norm(v):
    n=np.linalg.norm(v); 
    return v if n==0 else v/n

# =========================
#   BLENDER → MITSUBA
# =========================
def blender_cam_to_mitsuba_xyz(xb,yb,zb, variant="plugin"):
    """
    Recibe Euler XYZ (grados) de cámara en Blender y devuelve 3 ángulos XYZ
    para Mitsuba. 'variant':
      - 'canonical': la terna canónica (x,y,z)
      - 'plugin'   : rama equivalente que usa el add-on: (-x, y, 180 - z)
    """
    # 1) Rotación mundo Blender (intrínseco XYZ)
    RB = euler_to_R_intrinsic_xyz(xb,yb,zb)

    # 2) Base cámara en Blender (en mundo-B): right=+X, up=+Y, forward=-Z
    right_B   = _norm(RB @ np.array([1,0, 0],float))
    up_B      = _norm(RB @ np.array([0,1, 0],float))
    forward_B = _norm(RB @ np.array([0,0,-1],float))

    # 3) Mapea esa base a mundo Mitsuba
    right_M   = _norm(T @ right_B)
    up_M      = _norm(T @ up_B)
    forward_M = _norm(T @ forward_B)  # en Mitsuba la cámara mira +Z

    # 4) Matriz to_world de Mitsuba: columnas = (right, up, forward)
    RM = np.column_stack([right_M, up_M, forward_M])

    # 5) Descompone a (x,y,z) (intrínseco XYZ)
    x, y, z = R_to_euler_intrinsic_xyz(RM)

    if variant == "plugin":
        # Rama equivalente del plugin:
        x2 = wrap_deg(-x)
        y2 = wrap_deg( y)
        z2 = wrap_deg(180.0 - z)
        return (x2, y2, z2)
    else:
        return (x, y, z)

# =========================
#   MITSUBA → BLENDER
# =========================
def mitsuba_cam_to_blender_xyz(xm,ym,zm, assume_variant="plugin"):
    """
    Recibe Euler XYZ de Mitsuba (lo que hay en <rotate x/y/z>).
    Si 'assume_variant'='plugin', primero convierte a canónico usando (x,y,z) = (-xm, ym, 180 - zm),
    luego hace la conversión Mitsuba→Blender. Devuelve Euler XYZ en Blender.
    """
    if assume_variant == "plugin":
        # Volver a la rama canónica equivalente
        x_can = wrap_deg(-xm)
        y_can = wrap_deg( ym)
        z_can = wrap_deg(180.0 - zm)
        xm,ym,zm = x_can, y_can, z_can

    # 1) Rotación to_world en Mitsuba
    RM = euler_to_R_intrinsic_xyz(xm,ym,zm)

    # 2) Base cámara en Mitsuba
    right_M   = _norm(RM @ np.array([1,0,0],float))
    up_M      = _norm(RM @ np.array([0,1,0],float))
    forward_M = _norm(RM @ np.array([0,0,1],float))  # +Z local

    # 3) Llevar esa base a mundo Blender
    right_B   = _norm(Tinv @ right_M)
    up_B      = _norm(Tinv @ up_M)
    forward_B = _norm(Tinv @ forward_M)

    # 4) Reconstruir RB: columnas = (right, up, -forward) (en Blender mira -Z)
    RB = np.column_stack([right_B, up_B, -forward_B])

    # 5) Euler XYZ de Blender
    return R_to_euler_intrinsic_xyz(RB)
