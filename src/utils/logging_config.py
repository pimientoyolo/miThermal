"""
Configuración de logging para el proyecto
"""

import logging
import logging.config
from pathlib import Path
from ..config import get_output_path


def setup_logging(level: str = "INFO") -> None:
    """
    Configura el sistema de logging
    """
    
    # Crear directorio de logs
    log_dir = get_output_path("logs")
    log_file = log_dir / "app.log"
    
    # Configuración de logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(log_file),
                "mode": "a",
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "": {  # root logger
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }
    
    # Aplicar configuración
    logging.config.dictConfig(logging_config)
    
    # Log inicial
    logger = logging.getLogger(__name__)
    logger.info("Sistema de logging configurado")
    logger.info(f"Logs guardados en: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado
    """
    return logging.getLogger(name)
