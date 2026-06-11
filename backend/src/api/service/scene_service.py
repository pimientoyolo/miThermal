import logging

from fastapi import UploadFile

from fastapi import HTTPException
from fastapi.responses import FileResponse

import os
import glob
import shutil
import json
from src.utils.objects.objects import ObjectUtils
from src.utils.scene.parser import SceneParser
from src.utils.sensors.sensors import create_specfilm_bands
from src.utils.decorators import log_execution, handle_file_errors
from src.utils.helpers import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z, clamp_value
from src.atmosphere import create_homogeneous_medium, get_gas_manager
import numpy as np
from src.config import (
    PathManager, 
    OUTPUT_STATIC_DIR, 
    OUTPUT_STATIC_RESULT_DIR, 
    OUTPUT_SPD_DIR,
    OUTPUT_DIR,
    ASSETS_DIR,
    get_config_scene_dict,
    save_config_scene_dict
)
from src.api.service.base_services import BaseService, ZipHandler, FileHandler
from src.utils.cache import get_cache_manager

object_utils = ObjectUtils()
path_manager = PathManager

logger = logging.getLogger(__name__)


class SceneService(BaseService):
    
    def __init__(self):
        super().__init__()
        self.scene_parser = SceneParser()
        self.zip_handler = ZipHandler(self.logger)
        self.file_handler = FileHandler(self.logger)

    def load_scene(self, file: UploadFile):
        """Carga escena desde archivo ZIP usando ZipHandler."""
        # Validar que sea ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un archivo ZIP")

        # Guardar ZIP temporal
        scene_zip_path = path_manager.get_scene_zip_path()
        with open(scene_zip_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Limpiar y extraer con ZipHandler (reemplaza 20+ líneas)
        self.zip_handler.clear_and_extract(scene_zip_path, OUTPUT_STATIC_DIR)
        if os.path.exists(scene_zip_path):
            os.remove(scene_zip_path)
        
        # Recrear subdirectorios necesarios (asegurar que spds exista tras limpiar static)
        os.makedirs(OUTPUT_STATIC_RESULT_DIR, exist_ok=True)
        if os.path.exists(OUTPUT_SPD_DIR):
            shutil.rmtree(OUTPUT_SPD_DIR)
        os.makedirs(OUTPUT_SPD_DIR, exist_ok=True)
        air_file = os.path.join(OUTPUT_STATIC_DIR, "air.txt")
        air_default = os.path.join(ASSETS_DIR, "reference_data", "air.txt")
        if not os.path.exists(air_file) and os.path.exists(air_default):
            shutil.copy(air_default, air_file)

        # buscar y renombrar .xml
        xml_files = glob.glob(os.path.join(OUTPUT_STATIC_DIR, "*.xml"))
        logger.info(f"Archivos XML encontrados: {xml_files}")
        if len(xml_files) != 1:
            logger.error(f"Se esperaba exactamente 1 archivo XML, pero se encontraron {len(xml_files)}")
            raise HTTPException(status_code=400, detail="Debe haber exactamente un archivo XML en el ZIP")

        xml_file = xml_files[0]
        scene_rgb_path = path_manager.get_scene_path("rgb")
        os.rename(xml_file, scene_rgb_path)

        config_scene = {}
        T = 300

        scene_dict = self.get_dict_scene(scene_rgb_path)

        # cambiar el spp, width y height de la escena por defecto buscando por nombre
        defaults = scene_dict.get('scene', {}).get('default', [])
        if isinstance(defaults, list):
            for d in defaults:
                name = d.get("@name")
                if name == "spp":
                    d["@value"] = "256"
                elif name == "resx":
                    d["@value"] = "256"
                elif name == "resy":
                    d["@value"] = "256"
        elif isinstance(defaults, dict):
            name = defaults.get("@name")
            if name in ["spp", "resx", "resy"]:
                defaults["@value"] = "256"

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
        json_output_path = os.path.join(OUTPUT_DIR, "scene.json")
        with open(json_output_path, 'w') as json_file:
            json.dump(scene_dict, json_file, indent=2)
            
        logger.warning(f"guardado json en {json_output_path}")

        # revisar si se encuentran los archivos de los objetos y crear firmas
        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
            shapes = scene_dict["scene"]["shape"]
            config_scene["objects"] = {}  # inicializar como diccionario
            emissivity_file = path_manager.get_default_emissivity_path()
            logger.warning(f"Usando archivo de emisividad por defecto: {emissivity_file}")

            if isinstance(shapes, list):
                for shape in shapes:
                    id = shape["string"]["@value"]
                    # object_utils.save_default_emissivity(id)

                    config_scene["objects"][id] = {
                        "temperature": T,
                        "emissivity_file": emissivity_file
                    }

                    dir_file = os.path.join(OUTPUT_STATIC_DIR, id)
                    logger.info(f"Buscando archivo del objeto '{id}' en: {dir_file}")
                    if not os.path.exists(dir_file):
                        # Listar archivos disponibles para debug
                        available = os.listdir(OUTPUT_STATIC_DIR)
                        logger.error(f"No se encontró '{id}'. Archivos disponibles: {available}")
                        raise HTTPException(status_code=400, detail=f"No se encontró el archivo del objeto: {id}")

            elif isinstance(shapes, dict):
                id = shapes["string"]["@value"]
                # object_utils.save_default_emissivity(id)

                config_scene["objects"][id] = {
                    "temperature": T,
                    "emissivity_file": emissivity_file
                }

                dir_file = os.path.join(OUTPUT_STATIC_DIR, id)
                logger.info(f"Buscando archivo del objeto '{id}' en: {dir_file}")
                if not os.path.exists(dir_file):
                    # Listar archivos disponibles para debug
                    available = os.listdir(OUTPUT_STATIC_DIR)
                    logger.error(f"No se encontró '{id}'. Archivos disponibles: {available}")
                    raise HTTPException(status_code=400, detail=f"No se encontró el archivo del objeto: {id}")

        # air values
        t_air = 280

        wavelengths = np.linspace(10000, 12000, 10, endpoint=True, dtype=int)

        # Obtener longitudes de onda y atenuación desde archivo de referencia
        get_gas_manager().load_gas("air")
        num_bands = len(wavelengths)

        config_scene["air"] = {
            "temperature": t_air
        }
        rx = ry = rz = 0.0
        tx = ty = tz = 0.0

        # obtener sensor y transform
        sensor = scene_dict["scene"].get("sensor")
        if sensor is None or not isinstance(sensor, dict):
            logger.error("La escena no contiene un bloque 'sensor' válido")
            raise HTTPException(
                status_code=400,
                detail="La escena debe incluir un 'sensor' con parámetros de cámara"
            )

        transform = sensor.get("transform")

        # Asegurar que exista transform
        if transform is None or not isinstance(transform, dict):
            sensor["transform"] = {}
            transform = sensor["transform"]

        # obtner las rotaciones
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
        if not isinstance(translate, dict):
            translate = {"@value": "0 0 0"}
            transform["translate"] = translate

        value = translate.get("@value", "0 0 0")
        coords = value.split()
        if len(coords) != 3:
            logger.error(f"Formato inválido de translate: '{value}'")
            raise HTTPException(
                status_code=400,
                detail="El campo translate del sensor debe tener 3 coordenadas: 'x y z'"
            )

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
            "translate_y": -tz, # y mitsuba igual al z blender
            "translate_z": ty, # eje z igual al eje -y de blender
            "fov": fov
        }
        
        config_scene["num_bands"] = num_bands
        config_scene["wavelengths"] = wavelengths.tolist()

        # guardar cambios en scene.xml
        self.scene_parser.save_dict_as_xml(scene_dict, scene_rgb_path)

        # guardar config_scene como JSON
        config_scene_path = path_manager.get_config_scene_path()
        with open(config_scene_path, 'w') as f:
            json.dump(config_scene, f, indent=4)

        logger.warning(f"Escena cargada con éxito. Configuración guardada en {config_scene_path}")

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
        config_scene = get_config_scene_dict()

        num_bands = config_scene["num_bands"]
        wavelengths = config_scene["wavelengths"]

        # Validar que num_bands coincida con la longitud de wavelengths
        if len(wavelengths) != num_bands:
            raise HTTPException(
                status_code=400, 
                detail=f"El número de bandas ({num_bands}) no coincide con las longitudes ({len(wavelengths)})"
            )

        scene_rgb_path = path_manager.get_scene_path("rgb")
        if os.path.exists(scene_rgb_path):
            thermal_xml = path_manager.get_scene_path("thermal")
            shutil.copyfile(scene_rgb_path, thermal_xml)
            scene_dict = self.get_dict_scene(thermal_xml)

            # Cambiar el tipo de film y configurar sampler
            if scene_dict and "scene" in scene_dict and "sensor" in scene_dict["scene"]:
                if "film" in scene_dict["scene"]["sensor"]:
                    scene_dict["scene"]["sensor"]["film"]["@type"] = "specfilm"
                    
                    # Agregar bandas espectrales al film usando el helper para specfilm
                    scene_dict["scene"]["sensor"]["film"]["spectrum"] = create_specfilm_bands(wavelengths)
                
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
            # Eliminar referencias 'ref' en cada shape y forzar face_normals = true
            if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]
                
                def _force_face_normals(shape):
                    if "boolean" not in shape:
                        shape["boolean"] = {
                            "@name": "face_normals",
                            "@value": "true"
                        }
                    else:
                        bools = shape["boolean"]
                        if isinstance(bools, dict):
                            if bools.get("@name") == "face_normals":
                                bools["@value"] = "true"
                            else:
                                shape["boolean"] = [
                                    bools,
                                    {"@name": "face_normals", "@value": "true"}
                                ]
                        elif isinstance(bools, list):
                            has_fn = False
                            for b in bools:
                                if b.get("@name") == "face_normals":
                                    b["@value"] = "true"
                                    has_fn = True
                                    break
                            if not has_fn:
                                bools.append({"@name": "face_normals", "@value": "true"})

                if isinstance(shapes, list):
                    for shape in shapes:
                        if "ref" in shape:
                            del shape["ref"]
                        _force_face_normals(shape)
                elif isinstance(shapes, dict):
                    if "ref" in shapes:
                        del shapes["ref"]
                    _force_face_normals(shapes)

                # Agregar emisor diferente a cada shape
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        # para cada objeto
                        object_id = shape["string"]["@value"]
                        wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(object_id)
                        temperature = config_scene["objects"][object_id]["temperature"]
                        radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                        emission = radiance * emissivity
                        reflectance = 1 - np.array(emissivity)
                        
                        # Generar material y emisor con nombres significativos y obtener shasums
                        dict_reflectance, refl_sha = object_utils.create_reflectance_material(wavelengths_obj, reflectance, object_id)
                        dict_emission, emi_sha = object_utils.create_spectral_emitter(wavelengths_obj, emission, object_id)
                        
                        # Guardar info en configuración (Ruta absoluta para persistencia)
                        config_scene["objects"][object_id]["reflectance_shasum"] = refl_sha
                        config_scene["objects"][object_id]["reflectance_spd"] = str(OUTPUT_SPD_DIR / dict_reflectance["spectrum"]["@filename"].split('/')[-1])
                        config_scene["objects"][object_id]["emission_shasum"] = emi_sha
                        config_scene["objects"][object_id]["emission_spd"] = str(OUTPUT_SPD_DIR / dict_emission["spectrum"]["@filename"].split('/')[-1])
                        
                        shape["emitter"] = dict_emission
                        shape['bsdf'] = dict_reflectance
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape
                    object_id = shapes["string"]["@value"]
                    wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(object_id)
                    temperature = config_scene["objects"][object_id]["temperature"]
                    radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                    emission = radiance * emissivity
                    reflectance = 1 - np.array(emissivity)
                    
                    dict_reflectance, refl_sha = object_utils.create_reflectance_material(wavelengths_obj, reflectance, object_id)
                    dict_emission, emi_sha = object_utils.create_spectral_emitter(wavelengths_obj, emission, object_id)
                    
                    # Guardar info en configuración (Ruta absoluta para persistencia)
                    config_scene["objects"][object_id]["reflectance_shasum"] = refl_sha
                    config_scene["objects"][object_id]["reflectance_spd"] = str(OUTPUT_SPD_DIR / dict_reflectance["spectrum"]["@filename"].split('/')[-1])
                    config_scene["objects"][object_id]["emission_shasum"] = emi_sha
                    config_scene["objects"][object_id]["emission_spd"] = str(OUTPUT_SPD_DIR / dict_emission["spectrum"]["@filename"].split('/')[-1])
                    
                    shapes["emitter"] = dict_emission
                    shapes['bsdf'] = dict_reflectance

            # Guardar configuración con shasums
            save_config_scene_dict(config_scene)


            # Agregar medio homogéneo con coeficiente de extinción espectral
            # Crear valores de sigma_t para el medium (coeficiente de extinción)
            # Valores típicos para niebla en infrarrojo lejano
            t_air = config_scene["air"]["temperature"]
            wavelengths_air, sigma_t_values = get_gas_manager().load_gas("air")

            # La radiancia del aire se sumará analíticamente usando el mapa de profundidad
            # para evitar artefactos de aliasing y doble conteo en volpathmis.
            # Solo mantenemos el medium para que atenúe la radiación de los objetos.
            
            # Crear el medium usando la función de atmosphere
            medium_dict = create_homogeneous_medium(
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
        scene_rgb_path = path_manager.get_scene_path("rgb")
        depth_xml = path_manager.get_scene_path("depth")
        shutil.copyfile(scene_rgb_path, depth_xml)
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
        thermal_path = path_manager.get_scene_path("thermal")
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(thermal_path):
            self.prepare_thermal_scene()

        # Crear la escena de blackbody air copiando la escena térmica
        blackbody_air_xml = path_manager.get_scene_path("blackbody_air")
        shutil.copyfile(thermal_path, blackbody_air_xml)
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
        thermal_path = path_manager.get_scene_path("thermal")
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(thermal_path):
            self.prepare_thermal_scene()

        # Crear la escena de transmitancia blackbody air copiando la escena térmica
        transmittance_blackbody_air_xml = path_manager.get_scene_path("transmittance_blackbody_air")
        shutil.copyfile(thermal_path, transmittance_blackbody_air_xml)
        scene_dict = self.get_dict_scene(transmittance_blackbody_air_xml)

        config_scene = get_config_scene_dict()

        t_air = config_scene["air"]["temperature"]
        wavelengths_air, _ = get_gas_manager().load_gas("air")

        emission_air = object_utils.blackbody_radiance_nm(wavelengths_air, t_air)

        emitter_air_dict, _ = object_utils.create_spectral_emitter(wavelengths_air, emission_air)

        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]

                # Agregar emisor diferente a cada shape
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        
                        shape["emitter"] = emitter_air_dict
                        if "bsdf" in shape: del shape["bsdf"]
                        
                elif isinstance(shapes, dict):
                    # Para un solo shape

                    shapes["emitter"] = emitter_air_dict
                    if "bsdf" in shapes: del shapes['bsdf']

        # eliminar emisor de aire
        if scene_dict and "scene" in scene_dict and "emitter" in scene_dict["scene"]:
            del scene_dict["scene"]["emitter"]


        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, transmittance_blackbody_air_xml)

    def prepare_temperature_map(self):
        """
        Prepara la escena para renderización de la temperatura de los objetos.
        """
        thermal_path = path_manager.get_scene_path("thermal")
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(thermal_path):
            self.prepare_thermal_scene()

        # Crear la escena de transmitancia blackbody air copiando la escena térmica
        temperature_map_xml = path_manager.get_scene_path("temperature_map")
        shutil.copyfile(thermal_path, temperature_map_xml)
        scene_dict = self.get_dict_scene(temperature_map_xml)

        config_scene = get_config_scene_dict()

        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]

                # Agregar emisor de temperatura a cada objeto
                if isinstance(shapes, list):
                    for i, shape in enumerate(shapes):
                        id = shape["string"]["@value"]
                        temperature = float(config_scene["objects"][id]["temperature"])
                        
                        # Emitimos directamente el valor de la temperatura como radiancia constante
                        # Es CRITICO que el espectro tenga el nombre "radiance" para el plugin area
                        shape["emitter"] = {
                            "@type": "area",
                            "spectrum": {
                                "@name": "radiance",
                                "@type": "uniform",
                                "float": {"@name": "value", "@value": str(temperature)}
                            }
                        }
                        if "bsdf" in shape: del shape["bsdf"]

                elif isinstance(shapes, dict):
                    id = shapes["string"]["@value"]
                    temperature = float(config_scene["objects"][id]["temperature"])
                    shapes["emitter"] = {
                        "@type": "area",
                        "spectrum": {
                            "@name": "radiance",
                            "@type": "uniform",
                            "float": {"@name": "value", "@value": str(temperature)}
                        }
                    }
                    if "bsdf" in shapes: del shapes["bsdf"]

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

    def prepare_emissivity_map_scene(self):
        """
        Prepara la escena para renderizar el mapa de emisividad de los objetos pixel a pixel.
        Asigna el espectro de emisividad resampleado a las longitudes de onda de la escena
        como un emisor espectral.
        """
        thermal_path = path_manager.get_scene_path("thermal")
        # Verificar si existe la escena térmica, si no, crearla
        if not os.path.exists(thermal_path):
            self.prepare_thermal_scene()
            # Re-verificar después de intentar crearla
            if not os.path.exists(thermal_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"No se pudo crear o encontrar la escena térmica requerida: {thermal_path}"
                )

        # Crear la escena de mapa de emisividad copiando la escena térmica
        emissivity_map_xml = path_manager.get_scene_path("emissivity_map")
        shutil.copyfile(thermal_path, emissivity_map_xml)
        scene_dict = self.get_dict_scene(emissivity_map_xml)

        config_scene = get_config_scene_dict()
        wavelengths = config_scene.get("wavelengths", [])

        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
                shapes = scene_dict["scene"]["shape"]

                # Agregar emisor de emisividad a cada objeto
                def _add_emissivity_emitter(shape):
                    id = shape["string"]["@value"]
                    # Obtener emisividad espectral del objeto
                    wavelengths_obj, emiss_vals = object_utils.read_object_emissivity_file(id)
                    
                    # Interpolar a las longitudes de onda del sensor (incluyendo bandas de seguridad)
                    from src.utils.spectral.data_export import interpolate_spectral_data
                    wavelengths_scene = np.array(wavelengths)
                    emiss_resampled = interpolate_spectral_data(
                        np.array(wavelengths_obj),
                        np.array(emiss_vals),
                        wavelengths_scene
                    )
                    
                    # Generar emisor con el espectro de emisividad resampleado
                    dict_emission, emi_sha = object_utils.create_spectral_emitter(
                        wavelengths_scene,
                        emiss_resampled,
                        f"emissivity_{id}"
                    )
                    
                    shape["emitter"] = dict_emission
                    if "bsdf" in shape: del shape["bsdf"]

                if isinstance(shapes, list):
                    for shape in shapes:
                        _add_emissivity_emitter(shape)
                elif isinstance(shapes, dict):
                    _add_emissivity_emitter(shapes)

        # Cambiar el integrador a path para obtener el valor crudo sin sombras ni iluminación indirecta
        # Al eliminar BSDFs y usar emitters de área, obtenemos el valor del emisor
        if scene_dict and "scene" in scene_dict:
            scene_dict["scene"]["integrator"] = {
                "@type": "path",
                "integer": {
                    "@name": "max_depth",
                    "@value": "1"
                }
            }

        # Eliminar el medio atenuante para que no haya absorción
        if scene_dict and "scene" in scene_dict and "medium" in scene_dict["scene"]:
            del scene_dict["scene"]["medium"]

        if scene_dict and "scene" in scene_dict and "sensor" in scene_dict["scene"]:
            if "ref" in scene_dict["scene"]["sensor"]:
                del scene_dict["scene"]["sensor"]["ref"]

        # eliminar emisor de aire global si existe
        if scene_dict and "scene" in scene_dict and "emitter" in scene_dict["scene"]:
            del scene_dict["scene"]["emitter"]

        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, emissivity_map_xml)

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
        config_scene = get_config_scene_dict()

        thermal_path = path_manager.get_scene_path("thermal")
        scene_dict = self.get_dict_scene(thermal_path)
        
        # Verificar que el objeto existe en la configuración
        if object_id not in config_scene.get("objects", {}):
            raise HTTPException(
                status_code=404,
                detail=f"El objeto {object_id} no se encuentra en la configuración de la escena"
            )
        
        # Obtener las propiedades del objeto específico
        emissivity_file = config_scene["objects"][object_id]["emissivity_file"]
        wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(object_id)
        temperature = config_scene["objects"][object_id]["temperature"]


        object_found = False
        
        if scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]:
            shapes = scene_dict["scene"]["shape"]
            
            # Buscar y actualizar solo el objeto específico
            if isinstance(shapes, list):
                for i, shape in enumerate(shapes):
                    shape_id = shape["string"]["@value"]
                    if shape_id == object_id:
                        # Intentar obtener del cache persistente
                        cache = get_cache_manager()
                        result = cache.get(emissivity_file, temperature)
                        
                        if result is not None:
                            # Cargar de cache, pero verificar que los archivos .spd existan en disco
                            dict_refl, dict_emiss = result
                            refl_path = dict_refl.get("spectrum", {}).get("@filename")
                            emiss_path = dict_emiss.get("spectrum", {}).get("@filename")
                            
                            if refl_path and os.path.exists(refl_path) and emiss_path and os.path.exists(emiss_path):
                                dict_reflectance, dict_emission = result
                            else:
                                # Archivos borrados, forzar recalculación
                                result = None
                        
                        if result is None:
                            # Calcular y guardar en cache
                            radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                            emission = radiance * emissivity
                            reflectance = 1 - np.array(emissivity)
                            
                            dict_reflectance, refl_sha = object_utils.create_reflectance_material(wavelengths_obj, reflectance, object_id)
                            dict_emission, emi_sha = object_utils.create_spectral_emitter(wavelengths_obj, emission, object_id)
                            
                            # Actualizar config con shasums (Ruta absoluta para persistencia)
                            config_scene["objects"][object_id]["reflectance_shasum"] = refl_sha
                            config_scene["objects"][object_id]["reflectance_spd"] = str(OUTPUT_SPD_DIR / dict_reflectance["spectrum"]["@filename"].split('/')[-1])
                            config_scene["objects"][object_id]["emission_shasum"] = emi_sha
                            config_scene["objects"][object_id]["emission_spd"] = str(OUTPUT_SPD_DIR / dict_emission["spectrum"]["@filename"].split('/')[-1])
                            save_config_scene_dict(config_scene)
                            
                            cache.set(emissivity_file, temperature, dict_reflectance, dict_emission)

                        # Actualizar solo este objeto
                        shape["emitter"] = dict_emission
                        shape['bsdf'] = dict_reflectance
                        object_found = True
                        break

            elif isinstance(shapes, dict):
                # Para un solo shape
                shape_id = shapes["string"]["@value"]
                if shape_id == object_id:
                    # Intentar obtener del cache persistente
                    cache = get_cache_manager()
                    result = cache.get(emissivity_file, temperature)
                    
                    if result is not None:
                        # Cargar de cache
                        dict_reflectance, dict_emission = result
                    else:
                        # Calcular y guardar en cache
                        radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                        emission = radiance * emissivity
                        reflectance = 1 - np.array(emissivity)
                        
                        dict_reflectance, refl_sha = object_utils.create_reflectance_material(wavelengths_obj, reflectance, object_id)
                        dict_emission, emi_sha = object_utils.create_spectral_emitter(wavelengths_obj, emission, object_id)
                        
                        # Actualizar config con shasums
                        config_scene["objects"][object_id]["reflectance_shasum"] = refl_sha
                        config_scene["objects"][object_id]["reflectance_spd"] = dict_reflectance["spectrum"]["@filename"]
                        config_scene["objects"][object_id]["emission_shasum"] = emi_sha
                        config_scene["objects"][object_id]["emission_spd"] = dict_emission["spectrum"]["@filename"]
                        save_config_scene_dict(config_scene)
                        
                        cache.set(emissivity_file, temperature, dict_reflectance, dict_emission)
                    
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
            self.scene_parser.save_dict_as_xml(scene_dict, thermal_path)

    # Caché global en memoria

    def update_thermal_scene_shapes(self, object_ids: list[str]):
        """
        Actualiza todos los objetos en la escena térmica con sus propiedades actuales
        (temperatura + emisividad) en una sola pasada.
        Si la combinación (emisividad, temperatura) ya fue calculada, se carga de caché.
        """

        # Cargar configuración y escena térmica
        config_scene = get_config_scene_dict()
        thermal_path = path_manager.get_scene_path("thermal")
        scene_dict = self.get_dict_scene(thermal_path)

        if not (scene_dict and "scene" in scene_dict and "shape" in scene_dict["scene"]):
            raise HTTPException(status_code=404, detail="No se encontró la escena térmica")

        shapes = scene_dict["scene"]["shape"]

        # Normalizar: shapes puede ser dict o lista
        if isinstance(shapes, dict):
            shapes = [shapes]

        for shape in shapes:
            object_id = shape["string"]["@value"]

            if object_id not in config_scene["objects"]:
                continue  # ignorar shapes sin config

            if object_id not in object_ids:
                continue  # ignorar shapes no en la lista de IDs

            # Obtener propiedades del objeto
            emissivity_file = config_scene["objects"][object_id]["emissivity_file"]
            temperature = config_scene["objects"][object_id]["temperature"]

            # Intentar obtener del cache persistente
            cache = get_cache_manager()
            result = cache.get(emissivity_file, temperature)
            
            if result is not None:
                # Cargar de cache, pero verificar que existan archivos físicos
                dict_refl, dict_emiss = result
                refl_path = dict_refl.get("spectrum", {}).get("@filename")
                emiss_path = dict_emiss.get("spectrum", {}).get("@filename")
                
                if refl_path and os.path.exists(refl_path) and emiss_path and os.path.exists(emiss_path):
                    dict_reflectance, dict_emission = result
                else:
                    result = None

            if result is None:
                # Calcular y guardar en cache
                wavelengths_obj, emissivity = object_utils.read_object_emissivity_file(object_id)

                radiance = object_utils.blackbody_radiance_nm(wavelengths_obj, temperature)
                emission = radiance * emissivity
                reflectance = 1 - np.array(emissivity)

                dict_reflectance, refl_sha = object_utils.create_reflectance_material(wavelengths_obj, reflectance, object_id)
                dict_emission, emi_sha = object_utils.create_spectral_emitter(wavelengths_obj, emission, object_id)
                
                # Actualizar config (Ruta absoluta para persistencia)
                config_scene["objects"][object_id]["reflectance_shasum"] = refl_sha
                config_scene["objects"][object_id]["reflectance_spd"] = str(OUTPUT_SPD_DIR / dict_reflectance["spectrum"]["@filename"].split('/')[-1])
                config_scene["objects"][object_id]["emission_shasum"] = emi_sha
                config_scene["objects"][object_id]["emission_spd"] = str(OUTPUT_SPD_DIR / dict_emission["spectrum"]["@filename"].split('/')[-1])

                cache.set(emissivity_file, temperature, dict_reflectance, dict_emission)

            # Actualizar shape
            shape["emitter"] = dict_emission
            shape["bsdf"] = dict_reflectance

        # Guardar la escena actualizada y la configuración
        save_config_scene_dict(config_scene)
        self.scene_parser.save_dict_as_xml(scene_dict, thermal_path)

    def update_thermal_scene_air(self):
        """
        Actualiza las propiedades del aire en la escena térmica con sus nuevas propiedades.
        Refresca el espectro del film (sensor espectral) según las nuevas longitudes de onda.
        """
        config_scene = get_config_scene_dict()

        thermal_path = path_manager.get_scene_path("thermal")
        scene_dict = self.get_dict_scene(thermal_path)

        if not scene_dict or "scene" not in scene_dict:
            raise HTTPException(status_code=404, detail="Escena térmica no encontrada para actualizar aire")

        # === Actualizar film spectrum (extraído a helper) ===
        self.update_film_spectrum()

        # Datos de aire
        t_air = config_scene["air"]["temperature"]
        wavelengths, sigma_t_values = get_gas_manager().load_gas("air")

        # Recalcular emisión del aire
        emission_air = object_utils.blackbody_radiance_nm(wavelengths, t_air)
        emitter_air_dict, air_sha = object_utils.create_spectral_emitter(
            wavelengths, emission_air, "air", "constant"
        )
        
        # Actualizar config con shasums (Ruta absoluta para persistencia)
        config_scene["air"]["emission_shasum"] = air_sha
        config_scene["air"]["emission_spd"] = str(OUTPUT_SPD_DIR / emitter_air_dict["spectrum"]["@filename"].split('/')[-1])
        save_config_scene_dict(config_scene)

        scene_dict["scene"]["emitter"] = emitter_air_dict

        # Medium actualizado
        medium_dict = create_homogeneous_medium(
            wavelengths=wavelengths,
            sigma_t=sigma_t_values,
            medium_id="fog",
            g_value=0.95
        )
        scene_dict["scene"]["medium"] = medium_dict

        # Guardar la escena modificada como XML
        self.scene_parser.save_dict_as_xml(scene_dict, thermal_path)

    def update_film_spectrum(self) -> bool:
        """
        Actualiza el film del sensor para que use bandas espectrales acorde a 'wavelengths'.
        - Asegura el tipo 'specfilm' si ensure_specfilm=True

        Retorna True si se actualizó, False si no se encontró sensor/film.
        """
        config_scene = get_config_scene_dict()
        wavelengths = config_scene["wavelengths"]

        thermal_path = path_manager.get_scene_path("thermal")
        scene_dict = self.get_dict_scene(thermal_path)

        if not scene_dict or "scene" not in scene_dict:
            return False
        sensor = scene_dict["scene"].get("sensor")
        if not sensor or "film" not in sensor:
            return False

        film = sensor["film"]
        film["@type"] = "specfilm"
        film["spectrum"] = create_specfilm_bands(wavelengths)

        self.scene_parser.save_dict_as_xml(scene_dict, thermal_path)

        return True
    
    def _update_scene_camera_for_type(self, scene_type: str):
        scene_path = path_manager.get_scene_path(scene_type)

        if not os.path.exists(scene_path):
            # Intentar preparar la escena si no existe (excepto para RGB que debe ser la base)
            prep_methods = {
                "depth": self.prepare_depth_scene,
                "thermal": self.prepare_thermal_scene,
                "blackbody_air": self.prepare_blackbody_air_scene,
                "transmittance_blackbody_air": self.prepare_transmittance_blackbody_air_scene,
                "temperature_map": self.prepare_temperature_map,
                "emissivity_map": self.prepare_emissivity_map_scene,
            }
            
            if scene_type in prep_methods:
                self.logger.info(f"Escena '{scene_type}' no encontrada. Intentando preparar automáticamente...")
                try:
                    prep_methods[scene_type]()
                    # Re-verificar
                    if not os.path.exists(scene_path):
                        return
                except Exception as e:
                    self.logger.error(f"Error al preparar automáticamente '{scene_type}': {e}")
                    return
            else:
                return

        scene_dict = self.get_dict_scene(scene_path)

        if not scene_dict or "scene" not in scene_dict or "default" not in scene_dict["scene"]:
            raise HTTPException(
                status_code=400,
                detail=f"No se encontraron parámetros por defecto en la escena '{scene_type}'"
            )

        defaults = scene_dict["scene"]["default"]

        cam_cfg = get_config_scene_dict().get("camera", {})
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

        # 1) Actualizar valores por defecto (spp, resx, resy) buscando por nombre
        if isinstance(defaults, list):
            for d in defaults:
                name = d.get("@name")
                if name == "spp":
                    d["@value"] = str(spp)
                elif name == "resx":
                    d["@value"] = str(width)
                elif name == "resy":
                    d["@value"] = str(height)
        elif isinstance(defaults, dict):
            name = defaults.get("@name")
            if name == "spp":
                defaults["@value"] = str(spp)
            elif name == "resx":
                defaults["@value"] = str(width)
            elif name == "resy":
                defaults["@value"] = str(height)

        scene = scene_dict["scene"]
        sensor = scene.get("sensor")
        if sensor is None:
            raise HTTPException(
                status_code=400,
                detail=f"No se encontró sensor en la escena '{scene_type}'"
            )

        # 2) Actualizar sampler (spp) si está hardcodeado en el sensor
        sampler = sensor.get("sampler")
        if sampler:
            sintegers = sampler.get("integer")
            if isinstance(sintegers, list):
                for i in sintegers:
                    if i.get("@name") == "sample_count":
                        i["@value"] = str(spp)
            elif isinstance(sintegers, dict):
                if sintegers.get("@name") == "sample_count":
                    sintegers["@value"] = str(spp)

        # 3) Actualizar film (width, height) si están hardcodeados en el sensor
        film = sensor.get("film")
        if film:
            fintegers = film.get("integer")
            if isinstance(fintegers, list):
                for i in fintegers:
                    if i.get("@name") == "width":
                        i["@value"] = str(width)
                    elif i.get("@name") == "height":
                        i["@value"] = str(height)
            elif isinstance(fintegers, dict):
                if fintegers.get("@name") == "width":
                    fintegers["@value"] = str(width)
                elif fintegers.get("@name") == "height":
                    fintegers["@value"] = str(height)

        transform = sensor.get("transform")
        if transform is None:
            sensor["transform"] = {}
            transform = sensor["transform"]

        # Obtener target de la configuración si existe
        target_x = float(cam_cfg.get("target_x", 0.0))
        target_y = float(cam_cfg.get("target_y", 0.0))
        target_z = float(cam_cfg.get("target_z", 0.0))
        is_spherical = cam_cfg.get("is_spherical", False)
        
        # Obtener vector UP explícito si existe
        ux_cfg = cam_cfg.get("up_x")
        uy_cfg = cam_cfg.get("up_y")
        uz_cfg = cam_cfg.get("up_z")

        # Limpiar transformaciones antiguas si existen
        for key in ["rotate", "translate", "matrix", "lookat"]:
            if key in transform:
                del transform[key]

        # Si se usa modo esférico o hay un target explícito, usar lookat de Mitsuba
        # Esto es mucho más robusto para órbitas que usar ángulos Euler
        if is_spherical or any(v != 0.0 for v in [target_x, target_y, target_z]):
            # Aplicar cambio de base Blender -> Mitsuba: (x, y, z)_B -> (x, z, -y)_M
            origin_m = f"{tx} {tz} {-ty}"
            target_m = f"{target_x} {target_z} {-target_y}"
            
            # Determinar vector UP robusto
            if all(v is not None for v in [ux_cfg, uy_cfg, uz_cfg]):
                # Si hay un UP explícito en la config, convertir a Mitsuba
                # Blender (ux, uy, uz) -> Mitsuba (ux, uz, -uy)
                up_m = f"{ux_cfg} {uz_cfg} {-uy_cfg}"
            else:
                # Por defecto UP en Blender es (0,0,1)_B -> (0,1,0)_M
                up_m = "0 1 0"
                
                # Si estamos mirando exactamente hacia arriba o abajo, cambiar UP para evitar singularidad
                # Forward en Mitsuba
                fx, fy, fz = target_x - tx, target_z - tz, -(target_y - ty)
                fnorm = np.sqrt(fx*fx + fy*fy + fz*fz)
                if fnorm > 1e-6:
                    if abs(fy / fnorm) > 0.999:
                        # Si el eje de visión es paralelo al UP (0,1,0), usamos (0,0,1)
                        up_m = "0 0 1"
            
            transform["lookat"] = {
                "@origin": origin_m,
                "@target": target_m,
                "@up": up_m
            }
        else:
            # Si no hay target y no es esférico, usar el método de rotación Euler + Traslación
            angles = blender_cam_to_mitsuba_xyz(rx, ry, rz)
            transform["rotate"] = [
                {"@x": "1", "@angle": str(angles[0])},
                {"@y": "1", "@angle": str(angles[1])},
                {"@z": "1", "@angle": str(angles[2])}
            ]
            transform["translate"] = {
                "@value": f"{tx} {tz} {-ty}"
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
        else:
            sensor["float"] = {
                "@name": "fov",
                "@value": str(fov)
            }

        # Guardar la escena modificada como XML
        if scene_dict:
            self.scene_parser.save_dict_as_xml(scene_dict, scene_path)

    def update_scene_camera_rgb(self):
        self._update_scene_camera_for_type("rgb")

    def update_scene_camera_thermal(self):
        self._update_scene_camera_for_type("thermal")

    def update_scene_camera_all(self):
        """Actualiza la cámara en todas las escenas derivadas usadas en render."""
        scene_types = [
            "rgb",
            "depth",
            "thermal",
            "blackbody_air",
            "transmittance_blackbody_air",
            "temperature_map",
            "emissivity_map",
        ]
        for scene_type in scene_types:
            self._update_scene_camera_for_type(scene_type)

    def get_scene_mi_thermal(self) -> FileResponse:
        """Crea ZIP con escena actual usando ZipHandler."""
        
        path = str(OUTPUT_STATIC_DIR)
        zip_filename = path_manager.get_mithermal_scene_path() 

        # Una sola línea reemplaza todo el código de crear ZIP manualmente
        self.zip_handler.create_zip_from_directory(path, zip_filename)

        # Devolverlo como respuesta
        return FileResponse(zip_filename, media_type="application/zip", filename="miThermal.zip")
    
    def upload_mi_thermal_scene(self, file: UploadFile) -> None:
        """Carga escena miThermal desde ZIP usando ZipHandler."""
        path = str(OUTPUT_STATIC_DIR)
        zip_path = path_manager.get_mithermal_scene_path()

        # Guardar ZIP temporalmente
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Limpiar y extraer (una sola línea reemplaza 20+ líneas)
        self.zip_handler.clear_and_extract(zip_path, OUTPUT_STATIC_DIR)
        
        # Recrear subdirectorios y archivo air.txt de referencia
        os.makedirs(os.path.join(path, "result"), exist_ok=True)
        air_file = os.path.join(path, "air.txt")
        air_default = os.path.join(ASSETS_DIR, "reference_data", "air.txt")
        if not os.path.exists(air_file) and os.path.exists(air_default):
            shutil.copy(air_default, air_file)
    
    def get_suggested_mi_thermal_scene(self) -> list[str]:
        
        path = PathManager.get_default_scenes_dir()

        if not os.path.isdir(path):
            return []
        files = [os.path.basename(p) for p in glob.glob(os.path.join(path, "*.zip")) if os.path.isfile(p)]
        files.sort(key=str.lower)
        return files
    
    @log_execution()
    @handle_file_errors()
    def set_default_scene(self, scene_name: str) -> None:
        """
        Copia una escena por defecto al directorio de salida.
        Simplificada con ZipHandler.
        """
        default_scene_path = os.path.join(PathManager.get_default_scenes_dir(), scene_name)
        if not os.path.isfile(default_scene_path):
            raise HTTPException(status_code=404, detail=f"No se encontró la escena por defecto: {scene_name}")
        
        path = str(OUTPUT_STATIC_DIR)
        zip_filename = path_manager.get_mithermal_scene_path()

        # Copiar archivo ZIP
        shutil.copyfile(default_scene_path, zip_filename)

        # Limpiar y extraer con ZipHandler (reemplaza 25+ líneas)
        self.zip_handler.clear_and_extract(zip_filename, path)
        
        # Recrear archivo air.txt de referencia
        air_file = os.path.join(path, "air.txt")
        air_default = os.path.join(ASSETS_DIR, "reference_data", "air.txt")
        if not os.path.exists(air_file) and os.path.exists(air_default):
            shutil.copy(air_default, air_file)

    def generate_camera_animation(self, origin: list, end: list, tracked_point: list, num_steps: int = 30) -> list[dict]:
        """
        Genera una lista de configuraciones de cámara interpoladas para animación.
        
        Args:
            origin: Coordenadas [x, y, z] del punto inicial de la cámara
            end: Coordenadas [x, y, z] del punto final de la cámara
            tracked_point: Punto [x, y, z] al que la cámara siempre mira
            num_steps: Número de frames a generar (default: 30)
            
        Returns:
            Lista de diccionarios con configuración de cámara para cada frame
        """
        try:
            camera_frames = generate_camera_interpolation(origin, end, tracked_point, num_steps)
            self.logger.info(f"Generada interpolación de cámara con {num_steps} frames")
            return camera_frames
        except Exception as e:
            self.logger.error(f"Error al generar interpolación de cámara: {e}")
            raise HTTPException(status_code=500, detail=f"Error al generar interpolación: {str(e)}")

    def generate_camera_animation_spherical(
        self,
        start_theta: float,
        end_theta: float,
        start_azimuth: float,
        end_azimuth: float,
        tracked_point: list,
        radius: float = None,
        start_radius: float = None,
        end_radius: float = None,
        num_steps: int = 30,
        lock_azimuth_to_end: bool = False,
        theta_expr: str | None = None,
        azimuth_expr: str | None = None,
        radius_expr: str | None = None,
        auto_fov: bool = False,
        initial_fov: float | None = None,
    ) -> list[dict]:
        """Genera frames de cámara usando coordenadas esféricas alrededor de un objetivo."""
        try:
            camera_frames = generate_camera_interpolation_spherical(
                start_theta=start_theta,
                end_theta=end_theta,
                start_azimuth=start_azimuth,
                end_azimuth=end_azimuth,
                radius=radius,
                start_radius=start_radius,
                end_radius=end_radius,
                tracked_point=tracked_point,
                num_steps=num_steps,
                lock_azimuth_to_end=lock_azimuth_to_end,
                theta_expr=theta_expr,
                azimuth_expr=azimuth_expr,
                radius_expr=radius_expr,
                auto_fov=auto_fov,
                initial_fov=initial_fov,
            )
            self.logger.info(
                "Generada interpolación esférica con %s frames",
                num_steps,
            )
            return camera_frames
        except Exception as e:
            self.logger.error(f"Error al generar interpolación esférica: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error al generar interpolación esférica: {str(e)}",
            )

    def render_camera_path_preview_gif(self, camera_frames: list[dict]) -> str:
        """
        Renderiza un preview rápido de la trayectoria de cámara como GIF RGB.

        Estrategia:
        - Usa la mitad de los frames (submuestreo uniforme)
        - Reduce resolución y SPP para render rápido
        - Restaura configuración original de cámara al finalizar

        Args:
            camera_frames: Lista de configuraciones de cámara

        Returns:
            Ruta del GIF generado
        """
        from PIL import Image
        from src.api.service.render_service import RenderService
        from src.config import save_config_scene_dict

        if not camera_frames:
            raise HTTPException(
                status_code=400,
                detail="No hay frames para generar el preview"
            )

        render_service = RenderService()
        config_scene = get_config_scene_dict()

        original_camera = dict(config_scene.get("camera", {}))
        original_spp = int(original_camera.get("spp", 256))
        original_width = int(original_camera.get("width", 256))
        original_height = int(original_camera.get("height", 256))

        preview_frame_count = max(2, int(np.ceil(len(camera_frames) / 2)))
        preview_indices = np.linspace(
            0,
            len(camera_frames) - 1,
            preview_frame_count,
            dtype=int
        )
        preview_frames = [camera_frames[idx] for idx in preview_indices]

        preview_spp = max(4, min(32, original_spp // 8 if original_spp > 8 else 4))
        preview_width = max(160, original_width // 2)
        preview_height = max(120, original_height // 2)

        gif_frames: list[Image.Image] = []
        rgb_path = path_manager.get_result_path("rgb")
        gif_path = OUTPUT_DIR / "renders" / "camera_path_preview.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_scene["camera"].update({
                "spp": int(preview_spp),
                "width": int(preview_width),
                "height": int(preview_height),
            })
            save_config_scene_dict(config_scene)

            for frame_config in preview_frames:
                config_scene["camera"].update({
                    "translate_x": frame_config["translate_x"],
                    "translate_y": frame_config["translate_y"],
                    "translate_z": frame_config["translate_z"],
                    "rotate_x": frame_config["rotate_x"],
                    "rotate_y": frame_config["rotate_y"],
                    "rotate_z": frame_config["rotate_z"],
                    "target_x": frame_config.get("target_x", 0.0),
                    "target_y": frame_config.get("target_y", 0.0),
                    "target_z": frame_config.get("target_z", 0.0),
                    "is_spherical": frame_config.get("is_spherical", False),
                })
                save_config_scene_dict(config_scene)

                self.update_scene_camera_rgb()
                render_service.render_basic_scene()

                if os.path.exists(rgb_path):
                    frame = Image.open(rgb_path).convert("RGB")
                    gif_frames.append(frame.copy())

            if not gif_frames:
                raise HTTPException(
                    status_code=500,
                    detail="No se pudieron generar frames RGB para el preview"
                )

            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=120,
                loop=0,
            )
            self.logger.info(
                f"Preview GIF generado con {len(gif_frames)} frames: {gif_path}"
            )
            return str(gif_path)

        finally:
            config_scene["camera"].update({
                "spp": int(original_spp),
                "width": int(original_width),
                "height": int(original_height),
                "rotate_x": float(original_camera.get("rotate_x", 0.0)),
                "rotate_y": float(original_camera.get("rotate_y", 0.0)),
                "rotate_z": float(original_camera.get("rotate_z", 0.0)),
                "translate_x": float(original_camera.get("translate_x", 0.0)),
                "translate_y": float(original_camera.get("translate_y", 0.0)),
                "translate_z": float(original_camera.get("translate_z", 0.0)),
                "fov": float(original_camera.get("fov", 45.0)),
            })
            save_config_scene_dict(config_scene)
            try:
                self.update_scene_camera_all()
            except Exception as restore_err:
                self.logger.warning(
                    f"No se pudo restaurar XML de cámara tras preview: {restore_err}"
                )

    def render_camera_animation_sequence(
        self, 
        camera_frames: list[dict],
        spp: int = None,
        width: int = None,
        height: int = None,
        num_bands: int = None
    ) -> str:
        """
        Renderiza una secuencia completa de frames de animación.
        
        Para cada frame: actualiza la configuración de cámara, actualiza las escenas
        y renderiza todos los tipos (RGB, depth, thermal, etc.).
        Los resultados se guardan en output/renders/animation/frame_XXXX/
        
        Args:
            camera_frames: Lista de configuraciones de cámara generadas por generate_camera_animation
            spp: SPP opcional para el renderizado (sobrescribe config actual)
            width: Ancho opcional
            height: Alto opcional
            num_bands: Número de bandas opcional
            
        Returns:
            Ruta del archivo ZIP con todos los renders
        """
        from src.api.service.render_service import RenderService
        from src.config import save_config_scene_dict
        import zipfile
        
        render_service = RenderService()
        
        # Crear carpeta de animación
        animation_dir = OUTPUT_DIR / "renders" / "animation"
        animation_dir.mkdir(parents=True, exist_ok=True)
        
        # Limpiar carpeta anterior si existe
        if animation_dir.exists():
            shutil.rmtree(animation_dir)
        animation_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = len(camera_frames)
        self.logger.info(f"Iniciando renderizado de {total_frames} frames")
        
        # Obtener configuración actual para restaurar luego
        config_original = get_config_scene_dict().copy()
        config_scene = get_config_scene_dict()
        
        # Aplicar overrides si se proporcionan
        if spp is not None:
            config_scene["camera"]["spp"] = int(spp)
        if width is not None:
            config_scene["camera"]["width"] = int(width)
        if height is not None:
            config_scene["camera"]["height"] = int(height)
        
        # Si cambia el número de bandas, hay que regenerar longitudes de onda
        bands_changed = False
        if num_bands is not None and int(num_bands) != config_scene.get("num_bands"):
            from src.api.service.config_service import ConfigService
            cfg_service = ConfigService()
            w_min = min(config_scene["wavelengths"])
            w_max = max(config_scene["wavelengths"])
            cfg_service.update_wavelengths(w_min, w_max, int(num_bands))
            config_scene = get_config_scene_dict() # recargar
            bands_changed = True

        try:
            for idx, frame_config in enumerate(camera_frames):
                frame_num = idx + 1
                self.logger.info(f"Renderizando frame {frame_num}/{total_frames}")
                
                # Crear carpeta para este frame
                frame_dir = animation_dir / f"frame_{frame_num:04d}"
                frame_dir.mkdir(exist_ok=True)
                
                # Actualizar configuración de cámara (manteniendo spp/size seteados arriba)
                config_scene["camera"].update({
                    "translate_x": frame_config["translate_x"],
                    "translate_y": frame_config["translate_y"],
                    "translate_z": frame_config["translate_z"],
                    "rotate_x": frame_config["rotate_x"],
                    "rotate_y": frame_config["rotate_y"],
                    "rotate_z": frame_config["rotate_z"],
                    "target_x": frame_config.get("target_x", 0.0),
                    "target_y": frame_config.get("target_y", 0.0),
                    "target_z": frame_config.get("target_z", 0.0),
                    "is_spherical": frame_config.get("is_spherical", False),
                })
                save_config_scene_dict(config_scene)
                
                # Actualizar escenas XML con la nueva posición de cámara
                self.update_scene_camera_all()
                
                # Renderizar todos los tipos
                try:
                    # RGB
                    render_service.render_basic_scene()
                    rgb_src = path_manager.get_result_path("rgb")
                    if os.path.exists(rgb_src):
                        shutil.copy(rgb_src, frame_dir / "rgb.png")
                    
                    # Depth
                    render_service.render_depth_image()
                    depth_src = path_manager.get_result_path("depth")
                    if os.path.exists(depth_src):
                        shutil.copy(depth_src, frame_dir / "depth.npy")
                    
                    # Thermal
                    render_service.render_thermal_image()
                    thermal_src = path_manager.get_result_path("thermal")
                    if os.path.exists(thermal_src):
                        shutil.copy(thermal_src, frame_dir / "thermal.npy")
                    
                    # Thermal Raw (diagnóstico)
                    thermal_raw_src = path_manager.get_result_path("thermal_raw")
                    if os.path.exists(thermal_raw_src):
                        shutil.copy(thermal_raw_src, frame_dir / "thermal_raw.npy")
                    
                    # Blackbody Air
                    try:
                        blackbody_src = path_manager.get_result_path("blackbody_air")
                        if os.path.exists(blackbody_src):
                            shutil.copy(blackbody_src, frame_dir / "blackbody_air.npy")
                    except Exception as e:
                        self.logger.warning(
                            f"Frame {frame_num}: no se pudo copiar blackbody_air.npy: {e}"
                        )
                    
                    # Transmittance Blackbody Air
                    try:
                        trans_src = path_manager.get_result_path("transmittance_blackbody_air")
                        if os.path.exists(trans_src):
                            shutil.copy(trans_src, frame_dir / "transmittance_blackbody_air.npy")
                    except Exception as e:
                        self.logger.warning(
                            f"Frame {frame_num}: no se pudo copiar transmittance_blackbody_air.npy: {e}"
                        )
                    
                    # Contribution Air
                    try:
                        contrib_src = path_manager.get_result_path("contribution_blackbody_air")
                        if os.path.exists(contrib_src):
                            shutil.copy(contrib_src, frame_dir / "contribution_blackbody_air.npy")
                    except Exception as e:
                        self.logger.warning(
                            f"Frame {frame_num}: no se pudo copiar contribution_blackbody_air.npy: {e}"
                        )
                    
                    # Temperature Map
                    try:
                        render_service.render_temperature_map()
                        tmap_src = path_manager.get_result_path("temperature_map")
                        if os.path.exists(tmap_src):
                            shutil.copy(tmap_src, frame_dir / "temperature_map.npy")
                        else:
                            self.logger.warning(
                                f"Frame {frame_num}: no existe temperature_map.npy tras render"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Frame {frame_num}: no se pudo renderizar/copiar temperature_map.npy: {e}"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error renderizando frame {frame_num}: {e}")
                    continue
        finally:
            # Restaurar configuración original
            save_config_scene_dict(config_original)
            if bands_changed:
                from src.api.service.config_service import ConfigService
                cfg_service = ConfigService()
                w_min = min(config_original["wavelengths"])
                w_max = max(config_original["wavelengths"])
                cfg_service.update_wavelengths(w_min, w_max, config_original["num_bands"])
            self.update_scene_camera_all()

        # Crear ZIP con todos los frames
        zip_path = OUTPUT_DIR / "renders" / "animation.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for frame_dir in sorted(animation_dir.glob("frame_*")):
                for file in frame_dir.glob("*"):
                    arcname = f"{frame_dir.name}/{file.name}"
                    zipf.write(file, arcname)
        
        self.logger.info(f"Animación completada: {total_frames} frames renderizados en {zip_path}")
        return str(zip_path)


# --- Rotaciones básicas (usar helpers de geometría) ---
# Aliases para compatibilidad con código existente
Rx = rotation_matrix_x
Ry = rotation_matrix_y
Rz = rotation_matrix_z

# --- Euler intrínseco XYZ con vectores-columna: R = Rz(z) @ Ry(y) @ Rx(x) ---
def euler_to_R_intrinsic_xyz(x,y,z):
    return Rz(z) @ Ry(y) @ Rx(x)

# Alias para compatibilidad
clamp = clamp_value

def R_to_euler_intrinsic_xyz(R):
    """
    Descompone R = Rz(z) @ Ry(y) @ Rx(x) -> (x,y,z) en grados, normalizado a [-180,180).
    """
    y = np.arcsin(clamp(R[2,0]))
    cy = np.cos(y)
    eps = 1e-8
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
    n=np.linalg.norm(v) 
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


# =========================
#   INTERPOLACIÓN DE CÁMARA
# =========================
def calculate_look_at_blender(cam_pos: np.ndarray, target_pos: np.ndarray, up_global: np.ndarray = None):
    """
    Calcula los ángulos de Euler XYZ (en grados) para que una cámara en Blender
    (mirando hacia -Z local) apunte hacia target_pos.

    Args:
        cam_pos: Posición de la cámara [x, y, z]
        target_pos: Posición del objetivo [x, y, z]
        up_global: Vector 'arriba' de referencia (opcional)

    Returns:
        Tuple de ángulos Euler (x, y, z) en grados
    """
    if up_global is None:
        # Eje Z global en Blender es 'Arriba' por defecto
        up_global = np.array([0.0, 0.0, 1.0])

    # Vector hacia el objetivo
    forward = _norm(target_pos - cam_pos)

    # Manejar caso en el que estamos mirando directamente hacia arriba o abajo respecto al UP de referencia
    if abs(np.dot(forward, up_global)) > 0.999:
        # Si el UP es Z, cambiar a Y. Si es otra cosa, intentar Z o X.
        if abs(up_global[2]) > 0.9:
            up_global = np.array([0.0, 1.0, 0.0])
        else:
            up_global = np.array([0.0, 0.0, 1.0])
    right = _norm(np.cross(forward, up_global))
    up = np.cross(right, forward)
    
    # Matriz de rotación en Blender: columnas = (Right, Up, -Forward)
    R = np.column_stack([right, up, -forward])
    
    # Usar función existente para descomponer a (x,y,z) intrínsecos en grados
    euler_angles = R_to_euler_intrinsic_xyz(R)
    return euler_angles


def generate_camera_interpolation(origin, end, tracked_point, num_steps=30):
    """
    Genera una lista de diccionarios de configuración de cámara interpolada.
    
    Args:
        origin (list|tuple): Coordenadas [x, y, z] del inicio.
        end (list|tuple): Coordenadas [x, y, z] del final.
        tracked_point (list|tuple): Punto de interés [x, y, z] a mirar.
        num_steps (int): Cantidad de pasos (frames) a generar.
        
    Returns:
        List[dict]: Lista con las configuraciones (translate y rotate) para cada paso.
    """
    origin_np = np.array(origin, dtype=float)
    end_np = np.array(end, dtype=float)
    target_np = np.array(tracked_point, dtype=float)
    
    camera_frames = []
    
    for i in range(num_steps):
        # t va de 0.0 a 1.0
        t = i / max(1, (num_steps - 1))
        
        # Interpolación Lineal (LERP) de la posición
        current_pos = origin_np * (1.0 - t) + end_np * t
        
        # Calcular los ángulos Euler en Blender
        rot_x, rot_y, rot_z = calculate_look_at_blender(current_pos, target_np)
        
        frame_config = {
            "translate_x": float(current_pos[0]),
            "translate_y": float(current_pos[1]),
            "translate_z": float(current_pos[2]),
            "rotate_x": rot_x,
            "rotate_y": rot_y,
            "rotate_z": rot_z,
            "target_x": float(target_np[0]),
            "target_y": float(target_np[1]),
            "target_z": float(target_np[2]),
            "is_spherical": True
        }
        camera_frames.append(frame_config)
        
    return camera_frames


from src.utils.math_evaluator import safe_eval_t

def generate_camera_interpolation_spherical(
    start_theta,
    end_theta,
    start_azimuth,
    end_azimuth,
    radius=None,
    start_radius=None,
    end_radius=None,
    tracked_point=[0, 0, 0],
    num_steps=30,
    lock_azimuth_to_end=False,
    theta_expr=None,
    azimuth_expr=None,
    radius_expr=None,
    auto_fov=False,
    initial_fov=None,
):
    """Genera interpolación de cámara en coordenadas esféricas sobre un hemisferio con soporte opcional para funciones personalizadas."""
    if num_steps <= 0:
        raise ValueError("num_steps debe ser mayor que 0")
    
    # Determinar radios inicial y final
    s_rad = start_radius if start_radius is not None else radius
    e_rad = end_radius if end_radius is not None else radius
    
    if s_rad is None or e_rad is None:
        raise ValueError("Se debe proporcionar radius o (start_radius y end_radius)")

    # Preparar Auto-FOV si es necesario
    ref_size = None
    if auto_fov:
        # Si no se provee initial_fov, usar 45 por defecto
        base_fov = float(initial_fov) if initial_fov is not None else 45.0
        # Calcular tamaño aparente de referencia basado en el radio inicial
        # S = 2 * R * tan(FOV/2)
        ref_size = 2.0 * s_rad * np.tan(np.deg2rad(base_fov / 2.0))

    target_np = np.array(tracked_point, dtype=float)
    camera_frames = []

    for i in range(num_steps):
        t = i / max(1, (num_steps - 1))

        # 1) Interpolación base (lineal)
        t_base = t
        
        # 2) Aplicar expresiones personalizadas si existen
        # La expresión transforma 't' (0 a 1) en otro valor (normalmente 0 a 1)
        t_theta = safe_eval_t(theta_expr, t, t_base)
        t_azimuth = safe_eval_t(azimuth_expr, t, t_base)
        t_radius = safe_eval_t(radius_expr, t, t_base)

        # 3) Calcular ángulos finales
        theta_deg = float(start_theta * (1.0 - t_theta) + end_theta * t_theta)
        
        if lock_azimuth_to_end:
            azimuth_deg = float(end_azimuth)
        else:
            azimuth_deg = float(start_azimuth * (1.0 - t_azimuth) + end_azimuth * t_azimuth)
            
        current_radius = float(s_rad * (1.0 - t_radius) + e_rad * t_radius)

        theta_rad = np.deg2rad(theta_deg)
        azimuth_rad = np.deg2rad(azimuth_deg)

        offset = np.array(
            [
                current_radius * np.sin(theta_rad) * np.cos(azimuth_rad),
                current_radius * np.sin(theta_rad) * np.sin(azimuth_rad),
                current_radius * np.cos(theta_rad),
            ],
            dtype=float,
        )

        current_pos = target_np + offset
        # Por defecto usamos world Z [0,0,1]
        up_stable = np.array([0.0, 0.0, 1.0])
        
        # Si la cámara mira directamente hacia arriba/abajo (eje Z),
        # usamos world Y [0,1,0] para evitar la singularidad de look-at.
        forward = target_np - current_pos
        norm_f = np.linalg.norm(forward)
        if norm_f > 1e-6:
            forward = forward / norm_f
            if abs(np.dot(forward, up_stable)) > 0.99:
                up_stable = np.array([0.0, 1.0, 0.0])

        rot_x, rot_y, rot_z = calculate_look_at_blender(current_pos, target_np, up_global=up_stable)

        frame_config = {
            "translate_x": float(current_pos[0]),
            "translate_y": float(current_pos[1]),
            "translate_z": float(current_pos[2]),
            "rotate_x": float(rot_x),
            "rotate_y": float(rot_y),
            "rotate_z": float(rot_z),
            "target_x": float(target_np[0]),
            "target_y": float(target_np[1]),
            "target_z": float(target_np[2]),
            "up_x": float(up_stable[0]),
            "up_y": float(up_stable[1]),
            "up_z": float(up_stable[2]),
            "is_spherical": True,
            "theta": theta_deg,
            "azimuth": azimuth_deg,
            "radius": current_radius,
        }

        # Aplicar Auto-FOV
        if auto_fov and ref_size is not None:
            # FOV = 2 * arctan(S / (2 * R))
            # Usar arctan2 o asegurar que current_radius > 0
            safe_rad = max(1e-6, current_radius)
            new_fov_rad = 2.0 * np.arctan(ref_size / (2.0 * safe_rad))
            frame_config["fov"] = float(np.rad2deg(new_fov_rad))

        camera_frames.append(frame_config)

    return camera_frames
