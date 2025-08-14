"""
Pestaña de Exportación de Resultados
"""
import gradio as gr
from ..callbacks import export_results


def create_export_tab():
    """
    Crea la pestaña de exportación de resultados
    """
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
        export_btn.click(
            fn=export_results,
            inputs=[export_format, include_metadata],
            outputs=[export_status, download_files]
        )
        
        return {
            'export_format': export_format,
            'include_metadata': include_metadata,
            'export_btn': export_btn,
            'export_status': export_status,
            'download_files': download_files
        }
