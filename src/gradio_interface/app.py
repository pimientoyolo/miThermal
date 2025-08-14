import gradio as gr
import logging

from .layouts import create_main_layout, create_header
from .mitsuba_viewer.callbacks import create_mitsuba_viewer_interface
try:
    from config import get_config  # ejecución directa
except ImportError:
    from ..config import get_config  # como paquete

logger = logging.getLogger(__name__)
config = get_config()


def create_gradio_app() -> gr.Blocks:
    with create_main_layout() as app:
        create_header()
        with gr.Tabs():
            with gr.TabItem("Mitsuba Viewer"):
                create_mitsuba_viewer_interface()
    return app


def launch_gradio_app():
    try:
        logger.info("Iniciando Mitsuba Viewer UI...")
        app = create_gradio_app()
        gcfg = config["gradio"]
        app.launch(
            server_name=gcfg["host"],
            server_port=gcfg["port"],
            debug=gcfg["debug"],
            share=gcfg["share"],
        )
    except Exception as e:
        logger.error(f"Error al iniciar Gradio: {e}")
        raise


def create_app() -> gr.Blocks:  # compat
    logger.warning("create_app() deprecated; usar create_gradio_app().")
    return create_gradio_app()


def main():  # pragma: no cover
    launch_gradio_app()


if __name__ == "__main__":  # pragma: no cover
    main()
