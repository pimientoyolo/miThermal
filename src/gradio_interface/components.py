"""Componentes reutilizables
"""
from __future__ import annotations
import gradio as gr
from typing import Dict, Any

__all__ = [
	"build_upload_section",
	"build_object_viewer_section",
	"format_object_label",
]


def format_object_label(obj: Dict[str, Any]) -> str:
	"""Genera la etiqueta mostrada en el dropdown para un objeto de la escena.

	Reglas de negocio centralizadas:
	- id (type)
	- Añade 🎨 si tiene material
	- Añade 💡 si emite luz
	"""
	label = f"{obj['id']} ({obj.get('type','?')})"
	if obj.get("has_material"):
		label += " 🎨"
	if obj.get("has_emission"):
		label += " 💡"
	return label


def build_upload_section(show_server_path: bool = False) -> Dict[str, gr.components.Component]:
	"""Construye la sección de carga/render de escena.

	Args:
		show_server_path: Si se desea incluir input para cargar desde ruta servidor.
	Returns:
		Diccionario con referencias a componentes clave.
	"""
	with gr.Row():
		with gr.Column(scale=1):
			zip_file = gr.File(label="Escena Mitsuba (.zip)", file_types=[".zip"])
			upload_btn = gr.Button("📤 Subir Escena", variant="primary")
			render_btn = gr.Button("🖼️ Render RGB", variant="secondary")
			server_path = None
			load_btn = None
			if show_server_path:
				server_path = gr.Textbox(label="Ruta en servidor", placeholder="/ruta/a/escena.zip")
				load_btn = gr.Button("📥 Cargar del Servidor")
			scene_info = gr.Textbox(label="Info Escena", lines=6, interactive=False)
			scene_json = gr.JSON(label="Objetos (JSON)", visible=False)
		with gr.Column(scale=2):
			render_image = gr.Image(label="Render", type="pil", height=400)
	return {
		"zip_file": zip_file,
		"upload_btn": upload_btn,
		"render_btn": render_btn,
		"server_path": server_path,
		"load_btn": load_btn,
		"scene_info": scene_info,
		"scene_json": scene_json,
		"render_image": render_image,
	}


def build_object_viewer_section() -> Dict[str, gr.components.Component]:
	"""Construye la sección de visualización de objetos 3D."""
	with gr.Row():
		with gr.Column(scale=1):
			object_selector = gr.Dropdown(label="Objeto", choices=[], interactive=True)
			view_btn = gr.Button("👁️ Ver Objeto")
			object_info = gr.Textbox(label="Info Objeto", lines=15, interactive=False)
		with gr.Column(scale=2):
			model_viewer = gr.Model3D(label="Modelo 3D", height=600)
	return {
		"object_selector": object_selector,
		"view_btn": view_btn,
		"object_info": object_info,
		"model_viewer": model_viewer,
	}
