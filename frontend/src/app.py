#!/usr/bin/env python3
"""
Aplicación principal Gradio para simulaciones atmosféricas con Mitsuba
"""

import logging
from utils.logging_config import setup_logging
from gradio_interface import launch_gradio_app

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """
    Función principal de la aplicación - usa la nueva estructura modular
    """
    try:
        logger.info("Iniciando aplicación Gradio desde estructura modular...")
        launch_gradio_app()
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {e}")
        raise

if __name__ == "__main__":
    main()
