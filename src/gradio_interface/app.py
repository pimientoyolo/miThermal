"""
Aplicación principal de la interfaz Gradio modularizada
"""
import gradio as gr
import logging

# Imports locales
from .layouts import create_main_layout, create_header
from .tabs import (
    create_simulation_tab,
    create_spectral_analysis_tab,
    create_advanced_config_tab,
    create_export_tab
)
from ..config import get_config

# Configurar logging
logger = logging.getLogger(__name__)

# Cargar configuración
config = get_config()


def create_gradio_app() -> gr.Blocks:
    """
    Crea la aplicación principal de Gradio con estructura modular
    """
    
    with create_main_layout() as app:
        # Encabezado
        create_header()
        
        # Pestañas principales
        with gr.Tabs():
            # Pestaña de simulación
            simulation_components = create_simulation_tab()
            
            # Pestaña de análisis espectral  
            spectral_components = create_spectral_analysis_tab()
            
            # Pestaña de configuración avanzada
            config_components = create_advanced_config_tab()
            
            # Pestaña de exportación
            export_components = create_export_tab()
    
    return app


def launch_gradio_app():
    """
    Lanza la aplicación Gradio con la configuración especificada
    """
    try:
        logger.info("Iniciando aplicación Gradio modularizada...")
        
        # Crear la aplicación
        app = create_gradio_app()
        
        # Configurar y lanzar
        gradio_config = config["gradio"]
        app.launch(
            server_name=gradio_config["host"],
            server_port=gradio_config["port"],
            debug=gradio_config["debug"],
            share=gradio_config["share"]
        )
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación Gradio: {e}")
        raise


# Para compatibilidad hacia atrás
def create_app() -> gr.Blocks:
    """
    Función de compatibilidad - usar create_gradio_app() en su lugar
    """
    logger.warning("create_app() está deprecated. Usar create_gradio_app() en su lugar.")
    return create_gradio_app()


def main():
    """
    Función principal para ejecutar la aplicación
    """
    launch_gradio_app()


if __name__ == "__main__":
    main()
