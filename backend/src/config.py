"""
Configuración global del proyecto
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Crear directorios de salida si no existen
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "renders").mkdir(exist_ok=True)
(OUTPUT_DIR / "simulations").mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
(OUTPUT_DIR / "exports").mkdir(exist_ok=True)
(OUTPUT_DIR / "static").mkdir(exist_ok=True)
(OUTPUT_DIR / "static" / "result").mkdir(exist_ok=True)

# Copiar archivo air.txt por defecto si no existe
_air_output = OUTPUT_DIR / "static" / "air.txt"
_air_default = ASSETS_DIR / "reference_data" / "air.txt"
if not _air_output.exists() and _air_default.exists():
    shutil.copy(_air_default, _air_output)

# scenes dir
SCENE_DIR = str(OUTPUT_DIR / "static" / "scene.xml")
SCENE_THERMAL_DIR = str(OUTPUT_DIR / "static" / "scene_thermal.xml")
SCENE_DEPTH_DIR = str(OUTPUT_DIR / "static" / "scene_depth.xml")
SCENE_BLACKBODY_AIR = str(OUTPUT_DIR / "static" / "scene_blackbody_air.xml")
SCENE_TRANSMITTANCE_BLACKBODY_AIR = str(OUTPUT_DIR / "static" / "scene_transmittance_blackbody_air.xml")
SCENE_TEMPERATURE_MAP = str(OUTPUT_DIR / "static" / "scene_temperature_map.xml")

## OTHER FILES
OUTPUT_STATIC_DIR = OUTPUT_DIR / "static"
SCENE_ZIP = str(OUTPUT_DIR / "static" / "scene.zip")
CONFIG_SCENE = str(OUTPUT_DIR / "static" / "config_scene.json")
DEFAULT_EMISSIVITY_FILE = str(ASSETS_DIR / "materials" / "default.txt")  # (mantener por retrocompatibilidad con código existente)
AIR_ATTENUATION_FILE = str(OUTPUT_STATIC_DIR / "air.txt")
MITHERMAL_SCENE_FILE = str(OUTPUT_DIR / "miThermal.zip")
DEFAULT_SCENES_DIR = str(ASSETS_DIR / "mitsuba_scenes")
DEFAULT_EMISIVITY_DIR = str(ASSETS_DIR / "signatures")
DEFAULT_ATTENNUATION_DIR = str(ASSETS_DIR / "reference_data")

# resultados
OUTPUT_STATIC_RESULT_DIR = OUTPUT_DIR / "static" / "result"
IMAGE_DIR = str(OUTPUT_STATIC_RESULT_DIR / "rgb.png")
DEPTH_DIR = str(OUTPUT_STATIC_RESULT_DIR / "depth.npy")
THERMAL_DIR = str(OUTPUT_STATIC_RESULT_DIR / "thermal.npy")
BLACKBODY_AIR_DIR = str(OUTPUT_STATIC_RESULT_DIR / "blackbody_air.npy")
TRANSMITTANCE_BLACKBODY_AIR_DIR = str(OUTPUT_STATIC_RESULT_DIR / "transmittance_blackbody_air.npy")
CONTRIBUTION_BLACKBODY_AIR_DIR = str(OUTPUT_STATIC_RESULT_DIR / "contribution_blackbody_air.npy")
TEMPERATURE_MAP_DIR = str(OUTPUT_STATIC_RESULT_DIR / "temperature_map.npy")

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
