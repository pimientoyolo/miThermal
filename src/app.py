#!/usr/bin/env python3
"""
Aplicación principal Gradio para simulaciones atmosféricas con Mitsuba
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Imports del proyecto
from config import get_config, get_output_path
from gradio_interface.components import create_simulation_interface
from gradio_interface.callbacks import (
    run_atmospheric_simulation,
    load_spectral_data,
    export_results
)
from utils.logging_config import setup_logging

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Cargar configuración
config = get_config()

def create_app() -> gr.Blocks:
    """
    Crea la aplicación principal de Gradio
    """
    
    with gr.Blocks(
        title=config["gradio"]["title"],
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .output-image {
            max-height: 600px;
        }
        """
    ) as app:
        
        gr.HTML(f"""
        <h1 style="text-align: center; color: #2e86de;">
            🌍 {config["gradio"]["title"]}
        </h1>
        <p style="text-align: center; font-size: 18px;">
            {config["gradio"]["description"]}
        </p>
        """)
        
        with gr.Tab("Simulación Atmosférica"):
            simulation_interface = create_simulation_interface()
            
        with gr.Tab("Análisis Espectral"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>📊 Análisis de Firmas Espectrales</h3>")
                    
                    # Cargar datos espectrales
                    spectral_file = gr.File(
                        label="Archivo de datos espectrales (.npz)",
                        file_types=[".npz"]
                    )
                    
                    wavelength_range = gr.Slider(
                        minimum=1000,
                        maximum=20000,
                        value=[8000, 14000],
                        step=100,
                        label="Rango de longitudes de onda (nm)",
                        info="Rango espectral para análisis"
                    )
                    
                    analyze_btn = gr.Button("🔍 Analizar Espectro", variant="primary")
                    
                with gr.Column(scale=2):
                    spectral_plot = gr.Plot(label="Firma Espectral")
                    spectral_stats = gr.Dataframe(
                        label="Estadísticas Espectrales",
                        headers=["Propiedad", "Valor", "Unidades"]
                    )
        
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
        
        with gr.Tab("Exportar Resultados"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>💾 Exportación de Datos</h3>")
                    
                    export_format = gr.CheckboxGroup(
                        choices=["PNG", "EXR", "NPZ", "CSV", "JSON"],
                        value=["PNG", "CSV"],
                        label="Formatos de exportación"
                    )
                    
                    include_metadata = gr.Checkbox(
                        value=True,
                        label="Incluir metadatos"
                    )
                    
                    export_btn = gr.Button("📤 Exportar Resultados", variant="primary")
                    
                with gr.Column():
                    export_status = gr.Textbox(
                        label="Estado de exportación",
                        interactive=False
                    )
                    
                    download_files = gr.File(
                        label="Archivos generados",
                        file_count="multiple"
                    )
        
        # Configurar callbacks
        analyze_btn.click(
            fn=load_spectral_data,
            inputs=[spectral_file, wavelength_range],
            outputs=[spectral_plot, spectral_stats]
        )
        
        export_btn.click(
            fn=export_results,
            inputs=[export_format, include_metadata],
            outputs=[export_status, download_files]
        )
    
    return app

def main():
    """
    Función principal de la aplicación
    """
    try:
        logger.info("Iniciando aplicación Gradio...")
        
        # Crear la aplicación
        app = create_app()
        
        # Configurar y lanzar
        gradio_config = config["gradio"]
        app.launch(
            server_name=gradio_config["host"],
            server_port=gradio_config["port"],
            debug=gradio_config["debug"],
            share=gradio_config["share"]
        )
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {e}")
        raise

if __name__ == "__main__":
    main()
