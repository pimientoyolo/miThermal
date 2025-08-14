"""
Pestaña de Análisis Espectral
"""
import gradio as gr
from ..callbacks import load_spectral_data


def create_spectral_analysis_tab():
    """
    Crea la pestaña de análisis espectral
    """
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
        
        # Configurar callbacks
        analyze_btn.click(
            fn=load_spectral_data,
            inputs=[spectral_file, wavelength_range],
            outputs=[spectral_plot, spectral_stats]
        )
        
        return {
            'spectral_file': spectral_file,
            'wavelength_range': wavelength_range,
            'analyze_btn': analyze_btn,
            'spectral_plot': spectral_plot,
            'spectral_stats': spectral_stats
        }
