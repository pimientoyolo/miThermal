"""
Configuración global del proyecto
"""

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

SCENE_DIR = str(OUTPUT_DIR / "static" / "scene.xml")

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
