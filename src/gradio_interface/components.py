"""Componentes reutilizables
"""
from __future__ import annotations
import gradio as gr
from typing import Dict, Any

__all__ = [
	"build_upload_section",
	"build_object_viewer_section",
	"build_group_section",
	"build_object_group_section",
	"build_config_section",
	"build_unified_config_section",
	"build_visualization_section",
	"build_air_section",
	"build_spectrum_section",
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
			load_type = gr.Radio(label="Tipo de carga", choices=["Escena", "miTransfer"], value="Escena")
			zip_file = gr.File(label="Archivo (.zip)", file_types=[".zip"])
			upload_btn = gr.Button("📤 Subir", variant="primary")
			server_path = None
			load_btn = None
			if show_server_path:
				server_path = gr.Textbox(label="Ruta en servidor (solo Escena)", placeholder="/ruta/a/escena.zip")
				load_btn = gr.Button("📥 Cargar del Servidor")
			scene_info = gr.Textbox(label="Info Escena", lines=6, interactive=False)
			scene_json = gr.JSON(label="Objetos (JSON)", visible=False)
		with gr.Column(scale=2):
			render_image = gr.Image(label="Render RGB", type="pil", height=400)

	with gr.Row():
		with gr.Column(scale=3):
			default_suggest = gr.Dropdown(label="Escenas predeterminadas", choices=[], interactive=True)
		with gr.Column(scale=1):
			reload_default_btn = gr.Button("🔄 Recargar escenas predeterminadas", variant="secondary")
	with gr.Row():
		with gr.Column(scale=1):
			select_default_btn = gr.Button("✅ Seleccionar por defecto", variant="secondary")

	return {
		"load_type": load_type,
		"zip_file": zip_file,
		"upload_btn": upload_btn,
		"default_suggest": default_suggest,
		"reload_default_btn": reload_default_btn,
		"select_default_btn": select_default_btn,
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
			temperature_input = gr.Number(label="Temperatura (K)", value=None, precision=2)
			update_temp_btn = gr.Button("Actualizar Temperatura", variant="secondary")
			# Emisividad
			emissivity_plot = gr.Plot(label="Emisividad vs Longitud de Onda")
			emissivity_file = gr.File(label="Archivo Emisividad (tbs/txt)", file_types=[".txt", ".tbs"], interactive=True)
			update_emissivity_btn = gr.Button("Actualizar Emisividad", variant="secondary")
			# download_btn = gr.Button("💾 Descargar Archivo", variant="secondary")
			object_file = gr.File(label="Archivo Objeto", visible=False)
			object_info = gr.Textbox(label="Info Objeto", lines=15, interactive=False)
		with gr.Column(scale=2):
			model_viewer = gr.Model3D(label="Modelo 3D", height=600)
	return {
		"object_selector": object_selector,
		"view_btn": view_btn,
		"temperature_input": temperature_input,
		"update_temp_btn": update_temp_btn,
		"emissivity_plot": emissivity_plot,
		"emissivity_file": emissivity_file,
		"update_emissivity_btn": update_emissivity_btn,
		# "download_btn": download_btn,
		"object_file": object_file,
		"object_info": object_info,
		"model_viewer": model_viewer,
	}


def build_group_section() -> Dict[str, gr.components.Component]:
	"""Sección para operaciones masivas por grupo."""
	with gr.Row():
		with gr.Column(scale=1):
			group_level = gr.Radio(label="Nivel de grupo", choices=["Instancia", "Familia"], value="Instancia")
			group_selector = gr.Dropdown(label="Grupo", choices=[], interactive=True)
			group_members = gr.Dropdown(label="Miembros del Grupo", choices=[], multiselect=True, interactive=False)
			refresh_group_btn = gr.Button("🔄 Refrescar Grupos", variant="secondary")
			# Controles de propiedades
			group_temp_input = gr.Number(label="Temperatura Grupo (K)", value=None, precision=2)
			apply_temp_group_btn = gr.Button("Aplicar Temperatura a Grupo", variant="secondary")
			emissivity_file_group = gr.File(label="Archivo Emisividad Grupo", file_types=[".txt", ".tbs"], interactive=True)
			apply_emissivity_group_btn = gr.Button("Aplicar Emisividad a Grupo", variant="secondary")
			group_info = gr.Textbox(label="Info Grupo", lines=12, interactive=False)
		with gr.Column(scale=2):
			group_model_viewer = gr.Model3D(label="Preview Objeto del Grupo", height=600)
			group_emissivity_plot = gr.Plot(label="Emisividad Grupo (Preview)")
	return {
		"group_level": group_level,
		"group_selector": group_selector,
		"group_members": group_members,
		"refresh_group_btn": refresh_group_btn,
		"group_temp_input": group_temp_input,
		"apply_temp_group_btn": apply_temp_group_btn,
		"emissivity_file_group": emissivity_file_group,
		"apply_emissivity_group_btn": apply_emissivity_group_btn,
		"group_info": group_info,
		"group_model_viewer": group_model_viewer,
		"group_emissivity_plot": group_emissivity_plot,
	}


def build_object_group_section() -> Dict[str, gr.components.Component]:
	"""Sección unificada para trabajar por Objeto, Instancia o Familia."""
	with gr.Row():
		with gr.Column(scale=1):
			mode_radio = gr.Radio(label="Modo", choices=["Objeto", "Instancia", "Familia"], value="Objeto")
			selector = gr.Dropdown(label="Selecciona", choices=[], interactive=True)
			members = gr.Dropdown(label="Miembros", choices=[], multiselect=True, interactive=False)
			# Controles de propiedades
			temp_input = gr.Number(label="Temperatura (K)", value=None, precision=2)
			emissivity_file = gr.File(label="Archivo Emisividad", file_types=[".txt", ".tbs"], interactive=True)
			apply_update_btn = gr.Button("Actualizar Objeto(s)", variant="secondary")
			# Batch update objetos + atenuación
			# (Eliminado) batch_attenuation_file y batch_update_btn relacionados con atenuación masiva
			info_text = gr.Textbox(label="Info", lines=12, interactive=False)
		with gr.Column(scale=2):
			model_viewer = gr.Model3D(label="Vista 3D", height=600)
			emissivity_plot = gr.Plot(label="Emisividad (Preview)")
	return {
		"mode_radio": mode_radio,
		"selector": selector,
		"members": members,
		"temp_input": temp_input,
		"emissivity_file": emissivity_file,
		"apply_update_btn": apply_update_btn,
		# Campos eliminados: batch_attenuation_file, batch_update_btn
		"info_text": info_text,
		"model_viewer": model_viewer,
		"emissivity_plot": emissivity_plot,
	}


def build_config_section() -> Dict[str, gr.components.Component]:
	"""Sección para ver/editar configuración básica de cámara."""
	with gr.Row():
		with gr.Column(scale=1):
			config_info = gr.JSON(label="Config Actual")
			get_config_btn = gr.Button("Obtener Config", variant="secondary")
			camera_spp = gr.Slider(label="SPP (2^k)", minimum=1, maximum=13, step=1, value=4)
			camera_width = gr.Number(label="Ancho (px)", precision=0)
			camera_height = gr.Number(label="Alto (px)", precision=0)
			update_config_btn = gr.Button("Actualizar Config", variant="primary")
		with gr.Column(scale=2):
			config_notes = gr.Textbox(label="Notas", interactive=False, lines=8, value="Ajusta SPP, ancho y alto. Las longitudes de onda y num_bands se muestran en solo lectura.")
	return {
		"config_info": config_info,
		"get_config_btn": get_config_btn,
		"camera_spp": camera_spp,
		"camera_width": camera_width,
		"camera_height": camera_height,
		"update_config_btn": update_config_btn,
		"config_notes": config_notes,
	}


def build_unified_config_section() -> Dict[str, gr.components.Component]:
	"""Config unificada reorganizada:
	- Arriba: Diagrama de atenuación del aire (gr.Plot)
	- Abajo: 2 columnas
		• Izquierda (Config cámara): SPP, rotaciones y traslaciones, FOV
		• Derecha (Config general): λ min/max, bandas, ancho/alto, archivo de aire
	- Al final: JSON de configuración aplicada (una sola columna)
	"""
	# Arriba: Diagrama
	with gr.Row():
		air_plot = gr.Plot(label="Diagrama de atenuación del aire")

	# Zona media: dos columnas
	with gr.Row():
		# Izquierda: Config cámara
		with gr.Column(scale=1):
			gr.Markdown("### Config cámara")
			load_config_btn = gr.Button("Cargar Config", variant="secondary")
			camera_spp = gr.Slider(label="SPP (2^k)", minimum=1, maximum=13, step=1, value=4)
			rotate_x = gr.Number(label="Rotar X (deg)", precision=3)
			rotate_y = gr.Number(label="Rotar Y (deg)", precision=3)
			rotate_z = gr.Number(label="Rotar Z (deg)", precision=3)
			translate_x = gr.Number(label="Trasladar X", precision=6)
			translate_y = gr.Number(label="Trasladar Y", precision=6)
			translate_z = gr.Number(label="Trasladar Z", precision=6)
			fov = gr.Number(label="FOV (deg)", precision=3)

		# Derecha: Config general
		with gr.Column(scale=1):
			gr.Markdown("### Config general")
			wl_min = gr.Number(label="λ min (μm)", precision=0)
			wl_max = gr.Number(label="λ max (μm)", precision=0)
			bands = gr.Number(label="Número de bandas", precision=0)
			camera_width = gr.Number(label="Ancho (px)", precision=0)
			camera_height = gr.Number(label="Alto (px)", precision=0)
			air_temperature = gr.Number(label="Temperatura del aire (K)", precision=2)
			air_file = gr.File(label="Atenuación del Aire (txt)", file_types=[".txt", ".dat", ".csv"], interactive=True)
			with gr.Row():
				# air_suggest_btn = gr.Button("Sugerir atenuación", variant="secondary")
				air_suggest_list = gr.Dropdown(label="Archivo sugerido", choices=[], interactive=True)
				air_apply_suggest_btn = gr.Button("Aplicar sugerido", variant="secondary")

	# Botón aplicar y estado
	with gr.Row():
		apply_all_btn = gr.Button("Actualizar configuración", variant="primary")
		config_status = gr.Textbox(label="Estado Configuración", lines=4, interactive=False)

	# Abajo: JSON en una sola columna
	with gr.Row():
		config_info = gr.JSON(label="Configuración aplicada (JSON)")

	return {
		"air_plot": air_plot,
		"load_config_btn": load_config_btn,
		"camera_spp": camera_spp,
		"rotate_x": rotate_x,
		"rotate_y": rotate_y,
		"rotate_z": rotate_z,
		"translate_x": translate_x,
		"translate_y": translate_y,
		"translate_z": translate_z,
		"fov": fov,
		"wl_min": wl_min,
		"wl_max": wl_max,
		"bands": bands,
		"camera_width": camera_width,
		"camera_height": camera_height,
		"air_temperature": air_temperature,
		"air_file": air_file,
		# "air_suggest_btn": air_suggest_btn,
		"air_suggest_list": air_suggest_list,
		"air_apply_suggest_btn": air_apply_suggest_btn,
		"apply_all_btn": apply_all_btn,
		"config_status": config_status,
		"config_info": config_info,
	}


def build_visualization_section() -> Dict[str, gr.components.Component]:
	"""Sección para mostrar todos los renders juntos (galería)."""
	with gr.Row():
		# Botón a lo ancho
		run_sim_btn = gr.Button("▶️ Aplicar renderizado", variant="primary")
	with gr.Row():
		with gr.Column(scale=2, elem_id="viz_gallery_wrap"):
			gallery = gr.Gallery(label="Renders", show_label=True, columns=2, rows=3)
	with gr.Row():
		download_zip = gr.File(label="Descargar todos los renders (.zip)")
	return {
		"run_sim_btn": run_sim_btn,
		"gallery": gallery,
		"download_zip": download_zip,
	}


def build_air_section() -> Dict[str, gr.components.Component]:
	"""Sección para gestionar atenuación del aire."""
	with gr.Row():
		with gr.Column(scale=1):
			air_info = gr.Textbox(label="Atenuación (texto)", lines=10, interactive=False)
			get_air_btn = gr.Button("Obtener Atenuación", variant="secondary")
			air_file = gr.File(label="Archivo Atenuación (txt)", file_types=[".txt", ".dat", ".csv"], interactive=True)
			upload_air_btn = gr.Button("Actualizar Atenuación", variant="primary")
		with gr.Column(scale=2):
			air_plot = gr.Plot(label="Atenuación del Aire")
	return {
		"air_info": air_info,
		"get_air_btn": get_air_btn,
		"air_file": air_file,
		"upload_air_btn": upload_air_btn,
		"air_plot": air_plot,
	}


def build_spectrum_section() -> Dict[str, gr.components.Component]:
	"""Sección para configurar longitudes de onda y bandas."""
	with gr.Row():
		with gr.Column(scale=1):
			wl_min = gr.Number(label="λ min (μm)", precision=0)
			wl_max = gr.Number(label="λ max (μm)", precision=0)
			bands = gr.Number(label="Número de bandas", precision=0)
			update_btn = gr.Button("Actualizar Espectro", variant="primary")
		with gr.Column(scale=2):
			result = gr.JSON(label="Resultado")
	return {
		"wl_min": wl_min,
		"wl_max": wl_max,
		"bands": bands,
		"update_btn": update_btn,
		"result": result,
	}
