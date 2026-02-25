"""
Layouts y diseños para la interfaz Gradio
"""
import gradio as gr
try:
    from config import get_config  # ejecución directa
except ImportError:
    from ..config import get_config  # como paquete

# Cargar configuración
config = get_config()


def create_header():
    """
    Crea el encabezado principal de la aplicación
    """
    return gr.HTML(f"""
        <h1 style="text-align: center; color: #2e86de;">
            🌍 {config["gradio"]["title"]}
        </h1>
        <p style="text-align: center; font-size: 18px;">
            {config["gradio"]["description"]}
        </p>
    """)


def get_app_css():
    """
    Retorna el CSS personalizado para la aplicación
    """
    return """
    .gradio-container {
        max-width: 1400px !important;
    }
    .output-image {
        max-height: 600px;
    }
    """


def create_main_layout():
    """
    Crea el layout principal de la aplicación con Blocks
    """
    return gr.Blocks(
        title=config["gradio"]["title"],
        theme=gr.themes.Soft(),
        css=get_app_css()
    )
