"""
Configuración global del proyecto - Simplificada
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Directorios de salida
OUTPUT_STATIC_DIR = OUTPUT_DIR / "static"
OUTPUT_STATIC_RESULT_DIR = OUTPUT_STATIC_DIR / "result"
OUTPUT_SPD_DIR = OUTPUT_STATIC_DIR / "spds"
OUTPUT_ASSETS_DIR = OUTPUT_DIR / "assets"

# Directorios de assets
DEFAULT_SCENES_DIR = ASSETS_DIR / "mitsuba_scenes"
DEFAULT_EMISIVITY_DIR = ASSETS_DIR / "signatures"
DEFAULT_ATTENNUATION_DIR = ASSETS_DIR / "reference_data"

# Crear directorios necesarios
def _setup_directories():
    """Crea los directorios de salida necesarios"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "renders").mkdir(exist_ok=True)
    (OUTPUT_DIR / "simulations").mkdir(exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(exist_ok=True)
    (OUTPUT_DIR / "exports").mkdir(exist_ok=True)
    OUTPUT_STATIC_DIR.mkdir(exist_ok=True)
    OUTPUT_STATIC_RESULT_DIR.mkdir(exist_ok=True)
    OUTPUT_SPD_DIR.mkdir(exist_ok=True)
    OUTPUT_ASSETS_DIR.mkdir(exist_ok=True)
    
    # Copiar archivo air.txt por defecto si no existe
    _air_output = OUTPUT_STATIC_DIR / "air.txt"
    _air_default = DEFAULT_ATTENNUATION_DIR / "air.txt"
    if not _air_output.exists() and _air_default.exists():
        shutil.copy(_air_default, _air_output)

_setup_directories()


class PathManager:
    """Gestor centralizado de rutas del proyecto"""
    
    @staticmethod
    def get_scene_path(scene_type: str = "rgb") -> str:
        """
        Obtiene la ruta del archivo de escena según el tipo.
        
        Args:
            scene_type: Tipo de escena (rgb, thermal, depth, blackbody_air, 
                       transmittance_blackbody_air, temperature_map)
        
        Returns:
            Ruta del archivo de escena XML
        """
        scene_files = {
            "rgb": "scene.xml",
            "thermal": "scene_thermal.xml",
            "depth": "scene_depth.xml",
            "blackbody_air": "scene_blackbody_air.xml",
            "transmittance_blackbody_air": "scene_transmittance_blackbody_air.xml",
            "temperature_map": "scene_temperature_map.xml",
        }
        return str(OUTPUT_STATIC_DIR / scene_files.get(scene_type, "scene.xml"))
    
    @staticmethod
    def get_result_path(result_type: str) -> str:
        """
        Obtiene la ruta del archivo de resultado según el tipo.
        
        Args:
            result_type: Tipo de resultado (rgb, depth, thermal, blackbody_air, 
                        transmittance_blackbody_air, contribution_blackbody_air, temperature_map)
        
        Returns:
            Ruta del archivo de resultado
        """
        result_files = {
            "rgb": "rgb.png",
            "depth": "depth.npy",
            "thermal": "thermal.npy",
            "thermal_raw": "thermal_raw.npy",
            "blackbody_air": "blackbody_air.npy",
            "transmittance_blackbody_air": "transmittance_blackbody_air.npy",
            "contribution_blackbody_air": "contribution_blackbody_air.npy",
            "temperature_map": "temperature_map.npy",
        }
        return str(OUTPUT_STATIC_RESULT_DIR / result_files[result_type])
    
    @staticmethod
    def get_config_scene_path() -> str:
        """Obtiene la ruta del archivo de configuración de escena"""
        return str(OUTPUT_STATIC_DIR / "config_scene.json")
    
    @staticmethod
    def get_air_attenuation_path() -> str:
        """Obtiene la ruta del archivo de atenuación del aire"""
        return str(OUTPUT_STATIC_DIR / "air.txt")
    
    @staticmethod
    def get_scene_zip_path() -> str:
        """Obtiene la ruta del archivo ZIP de escena"""
        return str(OUTPUT_DIR / "scene_upload.zip")
    
    @staticmethod
    def get_default_emissivity_path() -> str:
        """Obtiene la ruta del archivo de emisividad por defecto"""
        return str(ASSETS_DIR / "materials" / "default.txt")
    
    @staticmethod
    def get_mithermal_scene_path() -> str:
        """Obtiene la ruta del archivo miThermal.zip"""
        return str(OUTPUT_DIR / "miThermal.zip")


# Retrocompatibilidad: mantener estas variables para código legacy
SCENE_DIR = PathManager.get_scene_path("rgb")
SCENE_THERMAL_DIR = PathManager.get_scene_path("thermal")
SCENE_DEPTH_DIR = PathManager.get_scene_path("depth")
SCENE_BLACKBODY_AIR = PathManager.get_scene_path("blackbody_air")
SCENE_TRANSMITTANCE_BLACKBODY_AIR = PathManager.get_scene_path("transmittance_blackbody_air")
SCENE_TEMPERATURE_MAP = PathManager.get_scene_path("temperature_map")

IMAGE_DIR = PathManager.get_result_path("rgb")
DEPTH_DIR = PathManager.get_result_path("depth")
THERMAL_DIR = PathManager.get_result_path("thermal")
BLACKBODY_AIR_DIR = PathManager.get_result_path("blackbody_air")
TRANSMITTANCE_BLACKBODY_AIR_DIR = PathManager.get_result_path("transmittance_blackbody_air")
CONTRIBUTION_BLACKBODY_AIR_DIR = PathManager.get_result_path("contribution_blackbody_air")
TEMPERATURE_MAP_DIR = PathManager.get_result_path("temperature_map")

CONFIG_SCENE = PathManager.get_config_scene_path()
AIR_ATTENUATION_FILE = PathManager.get_air_attenuation_path()
SCENE_ZIP = PathManager.get_scene_zip_path()
DEFAULT_EMISSIVITY_FILE = PathManager.get_default_emissivity_path()
MITHERMAL_SCENE_FILE = PathManager.get_mithermal_scene_path()

# Para strings
DEFAULT_SCENES_DIR = str(DEFAULT_SCENES_DIR)
DEFAULT_EMISIVITY_DIR = str(DEFAULT_EMISIVITY_DIR)
DEFAULT_ATTENNUATION_DIR = str(DEFAULT_ATTENNUATION_DIR)

# Configuración de Mitsuba
MITSUBA_CONFIG = {
    "variant": "cuda_ad_spectral",
    "spp": 1024,
    "max_depth": 8,
    "film_width": 512,
    "film_height": 512,
}

# Configuración de Gradio
GRADIO_CONFIG = {
    "port": 7860,
    "host": "0.0.0.0",
    "debug": True,
    "share": False,
    "title": "Simulador Atmosférico con Mitsuba",
    "description": "Simulación de efectos atmosféricos con rendering espectral",
}

# Configuración de datos espectrales
SPECTRAL_CONFIG = {
    "wavelength_range": (8000, 14000),  # LWIR en nm
    "num_bands": 49,
    "default_temperature": 5000,  # Kelvin
    "default_sigma_t": 0.01,
}

# Configuración de logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(OUTPUT_DIR / "logs" / "app.log"),
            "mode": "a",
        }
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file"]
        }
    }
}

def get_config() -> Dict[str, Any]:
    """Obtiene la configuración completa del proyecto"""
    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "src": SRC_DIR,
            "assets": ASSETS_DIR,
            "output": OUTPUT_DIR,
            "config": CONFIG_DIR,
            "notebooks": NOTEBOOKS_DIR,
            "tests": TESTS_DIR,
        },
        "mitsuba": MITSUBA_CONFIG,
        "gradio": GRADIO_CONFIG,
        "spectral": SPECTRAL_CONFIG,
        "logging": LOGGING_CONFIG,
    }

def get_mitsuba_variant() -> str:
    """Obtiene la variante de Mitsuba a usar"""
    return MITSUBA_CONFIG["variant"]

def get_spectral_range() -> tuple:
    """Obtiene el rango espectral configurado"""
    return SPECTRAL_CONFIG["wavelength_range"]

def get_output_path(subfolder: str = "") -> Path:
    """Obtiene la ruta de salida con subfolder opcional"""
    path = OUTPUT_DIR / subfolder if subfolder else OUTPUT_DIR
    path.mkdir(exist_ok=True)
    return path

def get_config_scene_dict() -> dict:
    cfg_path = Path(CONFIG_SCENE)
    if not cfg_path.exists():
        # crear estructura mínima por defecto
        minimal = {
            "objects": {},
            "air": {"temperature": 280},
            "camera": {
                "spp": 256, "width": 256, "height": 256,
                "rotate_x": 0.0, "rotate_y": 0.0, "rotate_z": 0.0,
                "translate_x": 0.0, "translate_y": 0.0, "translate_z": 0.0,
                "fov": 45.0
            },
            "num_bands": 0,
            "wavelengths": []
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, 'w') as fw:
            json.dump(minimal, fw, indent=4)
        return minimal
    with open(cfg_path, 'r') as f:
        return json.load(f)

def save_config_scene_dict(config_scene: dict) -> None:
    with open(CONFIG_SCENE, 'w') as f:
        json.dump(config_scene, f, indent=4)
