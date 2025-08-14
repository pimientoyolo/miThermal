"""
Componentes reutilizables para la interfaz Gradio
"""
import gradio as gr
import logging

logger = logging.getLogger(__name__)


def create_simulation_interface():
    """
    Crea la interfaz de simulación atmosférica
    
    Placeholder - implementar componentes específicos de simulación
    """
    with gr.Group():
        gr.HTML("<h3>🌍 Configuración de Simulación</h3>")
        
        with gr.Row():
            with gr.Column():
                scene_file = gr.File(
                    label="Archivo de escena comprimida de Mitsuba (.zip)",
                    file_types=[".zip"]
                )
                
                atmosphere_type = gr.Dropdown(
                    choices=["clear", "hazy", "foggy", "dusty"],
                    value="clear",
                    label="Tipo de atmósfera"
                )
                
            with gr.Column():
                simulation_params = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Parámetro de simulación"
                )
                
                run_btn = gr.Button("▶️ Ejecutar Simulación", variant="primary")
        
        with gr.Row():
            results_image = gr.Image(label="Resultado de simulación")
            results_info = gr.Textbox(
                label="Información de la simulación",
                lines=10,
                interactive=False
            )
    
    return {
        'scene_file': scene_file,
        'atmosphere_type': atmosphere_type,
        'simulation_params': simulation_params,
        'run_btn': run_btn,
        'results_image': results_image,
        'results_info': results_info
    }


def create_file_upload_component(label: str, file_types: list = None):
    """
    Crea un componente de subida de archivos reutilizable
    
    Args:
        label: Etiqueta del componente
        file_types: Lista de tipos de archivo permitidos
        
    Returns:
        Componente gr.File configurado
    """
    return gr.File(
        label=label,
        file_types=file_types or [],
        file_count="single"
    )


def create_parameter_slider(label: str, minimum: float, maximum: float, 
                          value: float, step: float = 1.0, info: str = None):
    """
    Crea un slider de parámetros reutilizable
    
    Args:
        label: Etiqueta del slider
        minimum: Valor mínimo
        maximum: Valor máximo  
        value: Valor por defecto
        step: Paso del slider
        info: Información adicional
        
    Returns:
        Componente gr.Slider configurado
    """
    return gr.Slider(
        minimum=minimum,
        maximum=maximum,
        value=value,
        step=step,
        label=label,
        info=info
    )


def create_status_display(label: str = "Estado", lines: int = 3):
    """
    Crea un display de estado reutilizable
    
    Args:
        label: Etiqueta del display
        lines: Número de líneas
        
    Returns:
        Componente gr.Textbox configurado para mostrar estado
    """
    return gr.Textbox(
        label=label,
        lines=lines,
        interactive=False,
        placeholder="Esperando..."
    )
