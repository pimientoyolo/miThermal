"""
Configuración global del proyecto
"""

import json
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

# Crear directorios de salida si no existen
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "renders").mkdir(exist_ok=True)
(OUTPUT_DIR / "simulations").mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
(OUTPUT_DIR / "exports").mkdir(exist_ok=True)
(OUTPUT_DIR / "static").mkdir(exist_ok=True)

# scenes dir
SCENE_DIR = str(OUTPUT_DIR / "static" / "scene.xml")
SCENE_THERMAL_DIR = str(OUTPUT_DIR / "static" / "scene_thermal.xml")
SCENE_DEPTH_DIR = str(OUTPUT_DIR / "static" / "scene_depth.xml")
SCENE_BLACKBODY_AIR = str(OUTPUT_DIR / "static" / "scene_blackbody_air.xml")
SCENE_TRANSMITTANCE_BLACKBODY_AIR = str(OUTPUT_DIR / "static" / "scene_transmittance_blackbody_air.xml")

## OTHER FILES
OUTPUT_STATIC_DIR = str(OUTPUT_DIR / "static")
SCENE_ZIP = str(OUTPUT_DIR / "static" / "scene.zip")
CONFIG_SCENE = str(CONFIG_DIR / "config_scene.json")
DEFAULT_EMITTIVITY_FILE = str(ASSETS_DIR / "materials" / "default.txt")
AIR_ATTENUATION_FILE = str(OUTPUT_DIR / "air.txt")

# resultados
OUTPUT_STATIC_RESULT_DIR = OUTPUT_DIR / "static" / "result"
IMAGE_DIR = str(OUTPUT_STATIC_RESULT_DIR / "rgb.png")
DEPTH_DIR = str(OUTPUT_STATIC_RESULT_DIR / "depth.npy")
THERMAL_DIR = str(OUTPUT_STATIC_RESULT_DIR / "thermal.npy")
BLACKBODY_AIR_DIR = str(OUTPUT_STATIC_RESULT_DIR / "blackbody_air.npy")
TRANSMITTANCE_BLACKBODY_AIR_DIR = str(OUTPUT_STATIC_RESULT_DIR / "transmittance_blackbody_air.npy")
CONTRIBUTION_BLACKBODY_AIR_DIR = str(OUTPUT_STATIC_RESULT_DIR / "contribution_blackbody_air.npy")

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
    with open(CONFIG_SCENE, 'r') as f:
        config_scene = json.load(f)
    return config_scene

def save_config_scene_dict(config_scene: dict) -> None:
    with open(CONFIG_SCENE, 'w') as f:
        json.dump(config_scene, f, indent=4)
