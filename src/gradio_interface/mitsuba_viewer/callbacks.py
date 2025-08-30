from __future__ import annotations
import logging
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from PIL import Image
import gradio as gr

from .api_client import MitsubaAPIClient
from .state import viewer_state
from ..components import build_upload_section, build_object_viewer_section

logger = logging.getLogger(__name__)
_api_client: MitsubaAPIClient | None = None

def get_client() -> MitsubaAPIClient:
    global _api_client
    if _api_client is None:
        _api_client = MitsubaAPIClient()
    return _api_client

def upload_zip_file(file_obj):
    """Sube ZIP, obtiene render y rellena dropdown con /object/suggest_objects.

    Returns (render_image, scene_info, object_selector_update, scene_json_data)
    """
    client = get_client()
    if file_obj is None:
        return None, "❌ No se seleccionó archivo", gr.update(choices=[], value=None), []
    try:
        result = client.upload_scene(file_obj.name)
        pil_image = None
        if result.get("status") == "ok" and result.get("image_base64"):
            try:
                pil_image = Image.open(BytesIO(base64.b64decode(result["image_base64"])))
            except Exception as e:
                logger.error(f"Decoding image error: {e}")
        # Sugerencias de objetos
        suggestions = client.suggest_objects() or []
        viewer_state.object_id_mapping = {}
        choices = []
        for s in suggestions:
            label = s.get('suggest') or s.get('id')
            choices.append(label)
            viewer_state.object_id_mapping[label] = s['id']
        info = f"✅ Escena cargada. Objetos sugeridos: {len(suggestions)}"
        return pil_image, info, gr.update(choices=choices, value=None), suggestions
    except Exception as e:
        logger.error(f"upload_zip_file error: {e}")
        return None, f"❌ Error: {e}", gr.update(choices=[], value=None), []

def load_server_scene(scene_path: str):
    """(Futuro) Carga desde ruta servidor - placeholder (endpoint actual no existe)."""
    return None, "⚠️ No implementado con endpoints actuales", gr.update(choices=[], value=None), []

def render_scene_rgb():
    """Fuerza un render y devuelve imagen actualizada sin cambiar lista objetos."""
    client = get_client()
    try:
        result = client.render_scene_rgb()
        if result.get("status") == "ok" and result.get("image_base64"):
            pil_image = Image.open(BytesIO(base64.b64decode(result["image_base64"])))
            info = "🖼️ Render actualizado"
            # No tocamos dropdown ni json -> devolver None para esos outputs gestionados
            return pil_image, info
        return None, "❌ Error render"
    except Exception as e:
        logger.error(f"render_scene_rgb error: {e}")
        return None, f"❌ Error: {e}"

def view_object_3d(choice: str):
    if not choice:
        return None, "Seleccione un objeto"
    oid = viewer_state.object_id_mapping.get(choice)
    if not oid:
        return None, "❌ ID de objeto no encontrado"
    client = get_client()
    try:
        tmp_dir = tempfile.mkdtemp()
        # Usar solo el nombre del archivo para evitar problemas con subcarpetas en temp
        tmp_base = Path(tmp_dir) / Path(oid).name
        downloaded = client.download_object(oid, str(tmp_base))
        if not downloaded:
            return None, f"❌ Error descargando objeto: {oid}"
        p = Path(downloaded)
        ext = p.suffix.lower()
        lines = [
            f"🎯 Objeto: {oid}",
            f"📄 Archivo: {p.name} ({ext})",
            f"💾 Tamaño: {p.stat().st_size} bytes",
        ]
        if ext == '.ply':
            lines.append("⚠️ PLY detectado (se convierte internamente si es necesario).")
        return str(p), "\n".join(lines)
    except Exception as e:
        logger.error(f"view_object_3d error: {e}")
        return None, f"❌ Error: {e}"

def create_mitsuba_viewer_interface():
    """Construye la interfaz principal usando componentes reutilizables."""
    with gr.Blocks(title="Mitsuba Scene Viewer") as interface:
        gr.HTML("""
        <h1 style='text-align:center;color:#2e86de;'>🎨 Visualizador de Escenas Mitsuba 3D</h1>
        <p style='text-align:center;'>Sube una escena Mitsuba (.zip), visualiza el render y explora objetos.</p>
        """)
        with gr.Tabs():
            with gr.Tab("📁 Cargar Escena"):
                upload_section = build_upload_section(show_server_path=False)
            with gr.Tab("🎯 Visualizar Objetos 3D"):
                object_section = build_object_viewer_section()
        upload_section['upload_btn'].click(
            fn=upload_zip_file,
            inputs=[upload_section['zip_file']],
            outputs=[
                upload_section['render_image'],
                upload_section['scene_info'],
                object_section['object_selector'],
                upload_section['scene_json'],
            ]
        )
        # Botón de render independiente
        upload_section['render_btn'].click(
            fn=render_scene_rgb,
            inputs=[],
            outputs=[upload_section['render_image'], upload_section['scene_info']]
        )
        if upload_section.get('load_btn') and upload_section.get('server_path'):
            upload_section['load_btn'].click(
                fn=load_server_scene,
                inputs=[upload_section['server_path']],
                outputs=[
                    upload_section['render_image'],
                    upload_section['scene_info'],
                    object_section['object_selector'],
                    upload_section['scene_json'],
                ]
            )
        object_section['view_btn'].click(
            fn=view_object_3d,
            inputs=[object_section['object_selector']],
            outputs=[object_section['model_viewer'], object_section['object_info']],
        )
    return interface
