"""
Pestaña de Configuración Avanzada
"""
import gradio as gr
from ...config import get_config

# Cargar configuración
config = get_config()


def create_advanced_config_tab():
    """
    Crea la pestaña de configuración avanzada
    """
    with gr.Tab("Configuración Avanzada"):
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>⚙️ Configuración de Mitsuba</h3>")
                
                mitsuba_variant = gr.Dropdown(
                    choices=["cuda_ad_spectral", "cuda_ad_rgb", "llvm_ad_spectral"],
                    value=config["mitsuba"]["variant"],
                    label="Variante de Mitsuba"
                )
                
                samples_per_pixel = gr.Slider(
                    minimum=64,
                    maximum=8192,
                    value=config["mitsuba"]["spp"],
                    step=64,
                    label="Muestras por píxel"
                )
                
                max_depth = gr.Slider(
                    minimum=2,
                    maximum=16,
                    value=config["mitsuba"]["max_depth"],
                    step=1,
                    label="Profundidad máxima de rayos"
                )
                
            with gr.Column():
                gr.HTML("<h3>🌡️ Parámetros Atmosféricos</h3>")
                
                temperature = gr.Slider(
                    minimum=200,
                    maximum=8000,
                    value=config["spectral"]["default_temperature"],
                    step=50,
                    label="Temperatura (K)"
                )
                
                sigma_t_scale = gr.Slider(
                    minimum=0.001,
                    maximum=1.0,
                    value=config["spectral"]["default_sigma_t"],
                    step=0.001,
                    label="Factor de extinción atmosférica"
                )
                
                atmospheric_model = gr.Dropdown(
                    choices=["clear", "hazy", "foggy", "dusty"],
                    value="clear",
                    label="Modelo atmosférico"
                )
        
        return {
            'mitsuba_variant': mitsuba_variant,
            'samples_per_pixel': samples_per_pixel,
            'max_depth': max_depth,
            'temperature': temperature,
            'sigma_t_scale': sigma_t_scale,
            'atmospheric_model': atmospheric_model
        }
