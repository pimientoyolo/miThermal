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
from ..components import build_upload_section, build_object_viewer_section, format_object_label

logger = logging.getLogger(__name__)
_api_client: MitsubaAPIClient | None = None

def get_client() -> MitsubaAPIClient:
    global _api_client
    if _api_client is None:
        _api_client = MitsubaAPIClient()
    return _api_client

def upload_zip_file(file_obj):
    client = get_client()
    if file_obj is None:
        return None, "❌ No se seleccionó archivo", gr.Dropdown(choices=[]), {}
    try:
        result = client.upload_scene(file_obj.name)
        pil_image = None
        if result.get("status") == "ok" and result.get("image_base64"):
            try:
                pil_image = Image.open(BytesIO(base64.b64decode(result["image_base64"])))
            except Exception as e:
                logger.error(f"Decoding image error: {e}")
        objects = []
        data = client.get_objects()
        if data.get("status") == "success":
            objects = data.get("objects", [])
        viewer_state.object_id_mapping = {}
        choices = []
        for obj in objects:
            label = format_object_label(obj)
            choices.append(label)
            viewer_state.object_id_mapping[label] = obj['id']
        if pil_image is not None:
            info = f"✅ Render recibido. Objetos: {len(objects)}"
        elif result.get("status") == "success":
            info = f"✅ Escena cargada (sin render). Objetos: {len(objects)}"
        else:
            return None, f"❌ Error: {result.get('detail','Error desconocido')}", gr.Dropdown(choices=[]), {}
        return pil_image, info, gr.Dropdown(choices=choices), objects
    except Exception as e:
        logger.error(f"upload_zip_file error: {e}")
        return None, f"❌ Error: {e}", gr.Dropdown(choices=[]), {}

def load_server_scene(scene_path: str):
    client = get_client()
    if not scene_path or not scene_path.strip():
        return None, "❌ Ingrese una ruta válida", gr.Dropdown(choices=[]), {}
    try:
        result = client.load_scene(scene_path.strip())
        if result.get("status") != "success":
            return None, f"❌ Error cargando escena: {result.get('detail','Error desconocido')}", gr.Dropdown(choices=[]), {}
        data = client.get_objects()
        if data.get("status") != "success":
            return None, f"❌ Error obteniendo objetos: {data.get('detail','Error desconocido')}", gr.Dropdown(choices=[]), {}
        objects = data.get("objects", [])
        viewer_state.object_id_mapping = {}
        choices = []
        for obj in objects:
            label = format_object_label(obj)
            choices.append(label)
            viewer_state.object_id_mapping[label] = obj['id']
        info = f"✅ Escena cargada. Objetos: {len(objects)}"
        return None, info, gr.Dropdown(choices=choices), objects
    except Exception as e:
        logger.error(f"load_server_scene error: {e}")
        return None, f"❌ Error: {e}", gr.Dropdown(choices=[]), {}

def view_object_3d(choice: str):
    if not choice:
        return None, "Seleccione un objeto"
    oid = viewer_state.object_id_mapping.get(choice)
    if not oid:
        return None, "❌ ID de objeto no encontrado"
    client = get_client()
    try:
        data = client.get_object(oid)
        if data.get("status") != "success":
            return None, f"❌ Error obteniendo objeto: {data.get('detail','Error desconocido')}"
        obj = data.get("object", {})
        tmp_dir = tempfile.mkdtemp()
        tmp_base = Path(tmp_dir) / oid
        downloaded = client.download_object(oid, str(tmp_base))
        if not downloaded:
            return None, f"❌ Error descargando objeto {oid}"
        p = Path(downloaded)
        ext = p.suffix.lower()
        lines = [
            f"🎯 Objeto: {obj['id']}",
            f"📦 Tipo: {obj['type']}",
            f"📄 Archivo: {obj.get('filename','')} ({ext})",
            f"💾 Tamaño: {p.stat().st_size} bytes",
        ]
        if obj.get('transform'):
            lines.append(f"🔄 Transformaciones: {len(obj['transform'])}")
        if obj.get('has_material'):
            lines.append(f"🎨 Material: {obj['bsdf'].get('type','?')}")
        if obj.get('has_emission'):
            lines.append(f"💡 Emisor: {obj['emitter'].get('type','?')}")
        if ext == '.ply':
            lines.append("⚠️ PLY detectado. Mejor convertir a OBJ si es binario.")
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
