"""
Pestaña de Simulación Atmosférica
"""
import gradio as gr


def create_simulation_tab():
    """
    Crea la pestaña de simulación atmosférica
    """
    with gr.Tab("Simulación Atmosférica"):
        # Crear directamente la interfaz para evitar imports circulares
        with gr.Group():
            gr.HTML("<h3>🌍 Configuración de Simulación</h3>")
            
            with gr.Row():
                with gr.Column():
                    scene_file = gr.File(
                        label="Archivo comprimido de escena de Mitsuba",
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
