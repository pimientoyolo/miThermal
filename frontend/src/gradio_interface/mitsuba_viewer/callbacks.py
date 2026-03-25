from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from ..components import (
    build_camera_interpolation_section,
    build_cache_management_section,
    build_spectral_plot_section,
)
from .api_client import MitsubaAPIClient
from .render_utils import normalize_to_uint8
from .state import viewer_state

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Cliente HTTP compartido
# --------------------------------------------------------------------------------------
_client: Optional[MitsubaAPIClient] = None


def get_client() -> MitsubaAPIClient:
    global _client
    if _client is None:
        _client = MitsubaAPIClient()
    return _client


# --------------------------------------------------------------------------------------
# Agrupación y estado
# --------------------------------------------------------------------------------------

def _compute_groups(objects: List[Dict]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Calcula grupos por instancia y familia a partir de una etiqueta estable.

    Reglas de base:
    - Usar obj['suggest'] si existe; si no, usar basename del id sin extensión.
    - Tomar el prefijo antes del primer '-' para eliminar sufijos de variante.
    - Instancia: si base coincide con '<prefijo>_<num>', instancia = ese base; familia = '<prefijo>'.
      Si no, y base termina en dígitos (p.ej., 'Sphere12'), instancia = base y familia = prefijo sin dígitos.
      En otro caso, instancia = familia = base.
    Retorna dicts mapping nombre_instancia/familia -> lista de object_id.
    """
    import os
    import re

    inst: Dict[str, List[str]] = {}
    fam: Dict[str, List[str]] = {}
    for obj in objects:
        oid = obj.get("id") or ""
        # Etiqueta estable para agrupar: suggest o basename(id)
        suggest = (obj.get("suggest") or "").strip()
        if suggest:
            label_src = suggest
        else:
            base_name = os.path.splitext(os.path.basename(oid))[0]
            label_src = base_name or oid

        # Normalizar base (antes de variantes '-')
        base = label_src.split("-", 1)[0]

        # Nueva lógica de familia: primer token antes de '_' si existe.
        # Ej: lamp_legup_Cube -> family='lamp'; mantiene familias más generales.
        if "_" in base:
            primary_family = base.split("_", 1)[0]
        else:
            primary_family = base

        # Instancia: usamos el nombre completo base (sin sufijo '-') para granularidad.
        instance = base

        # Si la heurística anterior (número al final) aporta un prefijo más corto, se ignora porque ya
        # queremos la familia principal por primer '_'. Sin embargo, para casos sin '_' mantenemos fallback.
        if "_" not in base:
            m = re.match(r"^(?P<prefix>.+?)(?P<num>\d+)$", base)
            if m and m.group("prefix"):
                primary_family = m.group("prefix")

        inst.setdefault(instance, []).append(oid)
        fam.setdefault(primary_family, []).append(oid)
    return inst, fam


def _summarize_scene_and_update_state() -> Tuple[str, Dict, List[str]]:
    client = get_client()
    data = client.get_objects()
    objs: List[Dict] = []
    if isinstance(data, dict):
        if isinstance(data.get("objects"), list):
            objs = data["objects"]
        elif data.get("status") == "success" and isinstance(data.get("data"), list):
            objs = data["data"]
    elif isinstance(data, list):
        objs = data

    import os
    labels: List[str] = []
    id_map: Dict[str, str] = {}
    for o in objs:
        oid = o.get("id") or ""
        # Usar 'suggest' como etiqueta visible; fallback a basename(id) sin extensión o al id
        suggest = (o.get("suggest") or "").strip()
        if suggest:
            label = suggest
        else:
            base = os.path.splitext(os.path.basename(oid))[0]
            label = base or oid or "?"
        id_map[label] = oid
        labels.append(label)
    labels_sorted = sorted(labels)
    viewer_state.object_id_mapping = id_map
    inst, fam = _compute_groups(objs)
    viewer_state.object_groups_instance = inst
    viewer_state.object_groups_family = fam
    viewer_state.selected_group_level = "instance"
    txt = f"Objetos: {len(objs)}\nInstancias: {len(inst)}\nFamilias: {len(fam)}"
    return txt, {"objects": objs}, labels_sorted


# --------------------------------------------------------------------------------------
# Carga de escenas
# --------------------------------------------------------------------------------------

def upload_zip_file(zip_file) -> Tuple[Optional[np.ndarray], str, Dict, List[str], List[str]]:
    if zip_file is None:
        return None, "❌ Sube un archivo .zip", {}, [], []
    path = getattr(zip_file, "name", None)
    if not path:
        return None, "❌ Archivo inválido", {}, [], []
    client = get_client()
    try:
        resp = client.upload_scene(path)
        # Intentar decodificar imagen si viene como base64
        img_pil = None
        try:
            if isinstance(resp, dict) and resp.get("image_base64"):
                import io
                import base64
                from PIL import Image
                img_bytes = base64.b64decode(resp["image_base64"])
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img_pil = None

        info_text, scene_json, labels = _summarize_scene_and_update_state()
        return img_pil, info_text, scene_json, labels, labels
    except Exception as e:
        return None, f"❌ Error subiendo escena: {e}", {}, [], []


def load_server_scene(scene_path: str) -> Tuple[Optional[np.ndarray], str, Dict, List[str], List[str]]:
    if not scene_path:
        return None, "❌ Ingresa la ruta en el servidor", {}, [], []
    client = get_client()
    try:
        resp = client.load_scene(scene_path)

        img_pil = None
        try:
            if isinstance(resp, dict) and resp.get("image_base64"):
                import io
                import base64
                from PIL import Image

                img_bytes = base64.b64decode(resp["image_base64"])
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                rgb_resp = client.session.get(f"{client.base_url}/render/rgb")
                if rgb_resp.ok and rgb_resp.content:
                    import io
                    from PIL import Image

                    img_pil = Image.open(io.BytesIO(rgb_resp.content)).convert("RGB")
        except Exception as preview_e:
            logger.warning(f"No se pudo cargar preview RGB tras load_scene: {preview_e}")

        info_text, scene_json, labels = _summarize_scene_and_update_state()
        return img_pil, info_text, scene_json, labels, labels
    except Exception as e:
        return None, f"❌ Error cargando escena: {e}", {}, [], []


# --------------------------------------------------------------------------------------
# Vista de objeto 3D
# --------------------------------------------------------------------------------------

def view_object_3d(choice_label: str):
    """Devuelve (ruta_modelo, info_text, ruta_archivo, temp_update, emisividad_plot)."""
    try:
        if not choice_label:
            return None, "Seleccione un objeto", None, gr.update(value=None), None
        oid = viewer_state.object_id_mapping.get(choice_label)
        if not oid:
            return None, f"❌ ID no mapeado para '{choice_label}'", None, gr.update(value=None), None
        client = get_client()
        obj_resp = client.get_object(oid)
        if obj_resp.get("status") != "success":
            return None, f"❌ Error obteniendo objeto {oid}", None, gr.update(value=None), None
        obj = obj_resp.get("object", {})
        # Descargar el archivo del objeto (cacheado por oid)
        if oid in viewer_state.object_file_cache:
            downloaded_path = viewer_state.object_file_cache[oid]
        else:
            tmp_dir = tempfile.mkdtemp()
            tmp_base = Path(tmp_dir) / oid
            downloaded_path = client.download_object(oid, str(tmp_base))
            if not downloaded_path:
                logger.error(f"view_object_3d: fallo download_object oid={oid}")
                return None, f"❌ Error descargando objeto {oid}", None, gr.update(value=None), None
            viewer_state.object_file_cache[oid] = downloaded_path
        p = Path(downloaded_path)

        ext = p.suffix.lower()
        obj_id = obj.get("id", oid)
        guessed_type = obj.get("type") or ("mesh" if ext in (".ply", ".obj") else "?")
        filename_display = obj.get("filename") or p.name
        lines = [
            f"🎯 Objeto: {obj_id}",
            f"📦 Tipo: {guessed_type}",
            f"📄 Archivo: {filename_display} ({ext})",
            f"💾 Tamaño: {p.stat().st_size} bytes",
            f"🗂️ Cache: {p}",
        ]
        if "temperature" in obj:
            lines.append(f"🌡️ Temp: {obj['temperature']}")
        if "emissivity" in obj and isinstance(obj["emissivity"], (list, tuple)):
            lines.append(f"📈 Emissivity bands: {len(obj['emissivity'])}")
        if obj.get("has_material"):
            lines.append(f"🎨 Material: {obj.get('bsdf', {}).get('type', '?')}")
        if obj.get("has_emission"):
            lines.append(f"💡 Emisor: {obj.get('emitter', {}).get('type', '?')}")
        temp_val = obj.get("temperature") if "temperature" in obj else None

        # Emisividad: si hay datos, graficar con Plotly
        emissivity_plot = None
        try:
            wavelengths = obj.get("wavelengths")
            reflection = obj.get("reflection")
            emissivity = obj.get("emissivity")
            if wavelengths and (reflection or emissivity):
                if reflection and not emissivity:
                    y = []
                    x = []
                    for wavelength, value in zip(wavelengths, reflection):
                        if wavelength is None or value is None:
                            continue
                        x.append(float(wavelength))
                        y.append(1 - (float(value) / 100.0))
                elif emissivity and wavelengths and len(emissivity) == len(wavelengths):
                    x = []
                    y = []
                    for wavelength, value in zip(wavelengths, emissivity):
                        if wavelength is None or value is None:
                            continue
                        x.append(float(wavelength))
                        y.append(float(value))
                else:
                    x = []
                    y = None
                if y:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='lines+markers',
                        name='Emisividad',
                        line=dict(color='#e67e22', width=2),
                        hovertemplate='<b>λ:</b> %{x:.1f}<br><b>Emisividad:</b> %{y:.4f}<extra></extra>'
                    ))
                    fig.update_layout(
                        title=f'Emisividad - {obj_id}',
                        xaxis_title='Longitud de onda (nm)',
                        yaxis_title='Emisividad (0-1)',
                        hovermode='closest',
                        template='plotly_white',
                        height=350,
                        showlegend=False,
                        yaxis=dict(range=[0, 1.05])
                    )
                    emissivity_plot = fig
        except Exception as plot_e:
            logger.warning(f"Fallo generando plot emisividad: {plot_e}")

        return str(p), "\n".join(lines), str(p), gr.update(value=temp_val), emissivity_plot
    except Exception as e:
        logger.error(f"view_object_3d error: {e}")
        return None, f"❌ Error: {e}", None, gr.update(value=None), None


# --------------------------------------------------------------------------------------
# Interfaz principal
# --------------------------------------------------------------------------------------

def create_mitsuba_viewer_interface():
    """Construye la interfaz principal usando componentes reutilizables."""
    with gr.Blocks(title="Mitsuba Scene Viewer") as interface:
        gr.HTML(
            """
        <style>
        /* Scroll para la galería de visualización */
        #viz_gallery_wrap { max-height: 70vh; overflow-y: auto; }
        /* Ajuste del botón superior */
        .gradio-container button { white-space: nowrap; }
        /* Tabs secundarias con estilo más discreto */
        .secondary-tabs { font-size: 0.9em; }
        </style>
        """
        )
        gr.HTML(
            """
            <h1 style='text-align:center;color:#2e86de;'>🎨 MiThermal</h1>
            <p style='text-align:center;color:#666;'>Simulador de Escenas Térmicas con Mitsuba</p>
            """
        )
        with gr.Tabs():
            # ==================== TAB 1: 🏠 ESCENA ====================
            with gr.Tab("🏠 Escena"):
                gr.Markdown("### Carga y Simulación de Escenas")
                
                # Sección de carga
                with gr.Row():
                    with gr.Column(scale=1):
                        load_type = gr.Radio(label="Tipo de carga", choices=["Escena", "miTransfer"], value="Escena")
                        zip_file = gr.File(label="Archivo (.zip)", file_types=[".zip"])
                        with gr.Row():
                            upload_btn = gr.Button("📤 Cargar Escena", variant="primary", size="lg")
                        with gr.Row():
                            default_suggest = gr.Dropdown(label="Escenas predeterminadas", choices=[], interactive=True)
                        with gr.Row():
                            reload_default_btn = gr.Button("🔄 Recargar", variant="secondary", scale=1)
                            select_default_btn = gr.Button("✅ Seleccionar", variant="secondary", scale=1)
                        scene_info = gr.Textbox(label="Estado de la Escena", lines=8, interactive=False)
                    
                    with gr.Column(scale=2):
                        render_image = gr.Image(label="Preview RGB", type="pil", height=400)
                
                gr.Markdown("---")
                gr.Markdown("### Simulación Completa")
                
                with gr.Row():
                    run_sim_btn = gr.Button("▶️ Ejecutar Simulación Completa", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2, elem_id="viz_gallery_wrap"):
                        gallery = gr.Gallery(label="Resultados de la Simulación", show_label=True, columns=2, rows=3)
                
                with gr.Row():
                    download_zip = gr.File(label="💾 Descargar Todos los Resultados (.zip)")
                
                # Referencias section combinadas
                upload_section = {
                    "load_type": load_type,
                    "zip_file": zip_file,
                    "upload_btn": upload_btn,
                    "default_suggest": default_suggest,
                    "reload_default_btn": reload_default_btn,
                    "select_default_btn": select_default_btn,
                    "scene_info": scene_info,
                    "render_image": render_image,
                    "scene_json": gr.JSON(visible=False),
                    "server_path": None,
                    "load_btn": None,
                }
                
                visualization_section = {
                    "run_sim_btn": run_sim_btn,
                    "gallery": gallery,
                    "download_zip": download_zip,
                }
            
            # ==================== TAB 2: 🎯 OBJETOS ====================
            with gr.Tab("🎯 Objetos"):
                with gr.Tabs(elem_classes=["secondary-tabs"]):
                    # Subtab: Explorador
                    with gr.Tab("🔍 Explorador"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                mode_radio = gr.Radio(label="Modo", choices=["Objeto", "Instancia", "Familia"], value="Objeto")
                                selector = gr.Dropdown(label="Selecciona", choices=[], interactive=True)
                                members = gr.Dropdown(label="Miembros", choices=[], multiselect=True, interactive=False)
                                gr.Markdown("#### Propiedades")
                                temp_input = gr.Number(label="Temperatura (K)", value=None, precision=2)
                                emissivity_file = gr.File(label="Archivo Emisividad", file_types=[".txt", ".tbs"], interactive=True)
                                apply_update_btn = gr.Button("✅ Aplicar Cambios", variant="primary")
                                info_text = gr.Textbox(label="Información del Objeto", lines=10, interactive=False)
                            
                            with gr.Column(scale=2):
                                model_viewer = gr.Model3D(label="Vista 3D", height=400)
                                emissivity_plot = gr.Plot(label="📊 Espectro de Emisividad")
                        
                        og_section = {
                            "mode_radio": mode_radio,
                            "selector": selector,
                            "members": members,
                            "temp_input": temp_input,
                            "emissivity_file": emissivity_file,
                            "apply_update_btn": apply_update_btn,
                            "info_text": info_text,
                            "model_viewer": model_viewer,
                            "emissivity_plot": emissivity_plot,
                        }
            
            # ==================== TAB 3: 📊 ANÁLISIS ====================
            with gr.Tab("📊 Análisis"):
                with gr.Tabs(elem_classes=["secondary-tabs"]):
                    # Subtab: Espectros
                    with gr.Tab("📈 Espectros"):
                        spectral_section = build_spectral_plot_section()
            
            # ==================== TAB 4: ⚙️ CONFIGURACIÓN ====================
            with gr.Tab("⚙️ Configuración"):
                with gr.Tabs(elem_classes=["secondary-tabs"]):
                    # Subtab: Cámara
                    with gr.Tab("📷 Cámara"):
                        gr.Markdown("### Parámetros de Renderizado")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### Calidad y Resolución")
                                camera_spp = gr.Slider(label="SPP (2^k)", minimum=1, maximum=13, step=1, value=4)
                                camera_width = gr.Number(label="Ancho (px)", precision=0, value=640)
                                camera_height = gr.Number(label="Alto (px)", precision=0, value=480)
                                fov = gr.Number(label="FOV (deg)", precision=3, value=45)
                            
                            with gr.Column(scale=1):
                                with gr.Tabs():
                                    with gr.Tab("Cartesiano"):
                                        gr.Markdown("#### Transformaciones")
                                        with gr.Row():
                                            rotate_x = gr.Number(label="Rotar X (°)", precision=3, value=0)
                                            rotate_y = gr.Number(label="Rotar Y (°)", precision=3, value=0)
                                            rotate_z = gr.Number(label="Rotar Z (°)", precision=3, value=0)
                                        with gr.Row():
                                            translate_x = gr.Number(label="Trasladar X", precision=6, value=0)
                                            translate_y = gr.Number(label="Trasladar Y", precision=6, value=0)
                                            translate_z = gr.Number(label="Trasladar Z", precision=6, value=0)
                                    
                                    with gr.Tab("Esférico (Ángulos)"):
                                        gr.Markdown("#### Coordenadas Esféricas")
                                        with gr.Row():
                                            theta = gr.Number(label="Zenital (theta) [0-180]", precision=3)
                                            phi = gr.Number(label="Azimutal (phi) [0-360]", precision=3)
                                            radius = gr.Number(label="Radio", precision=3)
                                        gr.Markdown("#### Punto Objetivo (Target)")
                                        with gr.Row():
                                            target_x = gr.Number(label="Target X", precision=3, value=0.0)
                                            target_y = gr.Number(label="Target Y", precision=3, value=0.0)
                                            target_z = gr.Number(label="Target Z", precision=3, value=0.0)
                        
                        gr.Markdown("---")
                        gr.Markdown("### Interpolación de Cámara (Animaciones)")
                        camera_interp_section = build_camera_interpolation_section()
                    
                    # Subtab: Espectro
                    with gr.Tab("🌈 Espectro"):
                        gr.Markdown("### Configuración Espectral")
                        with gr.Row():
                            with gr.Column(scale=1):
                                wl_min = gr.Number(label="λ mínima (nm)", precision=0, value=8000)
                                wl_max = gr.Number(label="λ máxima (nm)", precision=0, value=12000)
                                bands = gr.Number(label="Número de bandas", precision=0, value=50)
                                spectrum_apply_btn = gr.Button("✅ Aplicar Configuración Espectral", variant="primary")
                            
                            with gr.Column(scale=2):
                                gr.Markdown("#### Información")
                                gr.Markdown("""
                                - **Rango espectral**: Define el intervalo de longitudes de onda para la simulación
                                - **Bandas**: Mayor número = mayor precisión pero más tiempo de cómputo
                                - **Rango típico IR térmico**: 8000-14000 nm
                                """)
                                spectrum_status = gr.Textbox(label="Estado", lines=3, interactive=False)
                    
                    # Subtab: Atmósfera
                    with gr.Tab("🌫️ Atmósfera"):
                        gr.Markdown("### Atenuación Atmosférica")
                        with gr.Row():
                            air_plot = gr.Plot(label="Diagrama de Atenuación del Aire")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                air_temperature = gr.Number(label="Temperatura del aire (K)", precision=2, value=300)
                                air_file = gr.File(label="Archivo de Atenuación (.txt)", file_types=[".txt", ".dat", ".csv"], interactive=True)
                                with gr.Row():
                                    air_suggest_list = gr.Dropdown(label="Archivos sugeridos", choices=[], interactive=True)
                                    air_apply_suggest_btn = gr.Button("✅ Aplicar", variant="primary")
                            
                            with gr.Column(scale=1):
                                gr.Markdown("#### Gases disponibles")
                                gr.Markdown("""
                                - **air.txt**: Atmósfera estándar
                                - **H2O.txt**: Vapor de agua
                                - **CO2.txt**: Dióxido de carbono
                                - **O3.txt**: Ozono
                                - **CH4.txt**: Metano
                                """)
                                atm_status = gr.Textbox(label="Estado", lines=4, interactive=False)
                    
                    # Subtab: Avanzado
                    with gr.Tab("🔧 Avanzado"):
                        gr.Markdown("### Aplicar Toda la Configuración")
                        with gr.Row():
                            load_config_btn = gr.Button("📥 Cargar Config Actual", variant="secondary")
                            export_cam_btn = gr.Button("📤 Exportar Cámara", variant="secondary")
                            import_cam_btn = gr.Button("📥 Importar Cámara", variant="secondary")
                            apply_all_btn = gr.Button("✅ Aplicar Toda la Config", variant="primary", size="lg")
                        
                        with gr.Row():
                            config_status = gr.Textbox(label="Estado de Configuración", lines=4, interactive=False)
                        
                        with gr.Row():
                            config_info = gr.JSON(label="Configuración Completa (JSON)")
                        
                        gr.Markdown("---")
                        gr.Markdown("### Gestión de Cache")
                        cache_section = build_cache_management_section()
                
                # Unified config section para compatibilidad con callbacks
                config_section = {
                    "camera_spp": camera_spp,
                    "camera_width": camera_width,
                    "camera_height": camera_height,
                    "rotate_x": rotate_x,
                    "rotate_y": rotate_y,
                    "rotate_z": rotate_z,
                    "translate_x": translate_x,
                    "translate_y": translate_y,
                    "translate_z": translate_z,
                    "theta": theta,
                    "phi": phi,
                    "radius": radius,
                    "target_x": target_x,
                    "target_y": target_y,
                    "target_z": target_z,
                    "fov": fov,
                    "wl_min": wl_min,
                    "wl_max": wl_max,
                    "bands": bands,
                    "spectrum_apply_btn": spectrum_apply_btn,
                    "spectrum_status": spectrum_status,
                    "air_temperature": air_temperature,
                    "air_file": air_file,
                    "air_suggest_list": air_suggest_list,
                    "air_apply_suggest_btn": air_apply_suggest_btn,
                    "air_plot": air_plot,
                    "atm_status": atm_status,
                    "load_config_btn": load_config_btn,
                    "export_cam_btn": export_cam_btn,
                    "import_cam_btn": import_cam_btn,
                    "apply_all_btn": apply_all_btn,
                    "config_status": config_status,
                    "config_info": config_info,
                }

        # Visualización: ejecutar todos los renders y mostrarlos en galería
        def run_simulation_cb():
            """Renderiza todos los mapas necesarios y crea un ZIP en memoria con los .npy.
            - Evita archivos temporales intermedios para cada .npy
            - Verifica integridad cargando los arrays desde bytes antes de empaquetar
            - Devuelve una ruta a un único ZIP temporal por compatibilidad con gr.File
            """
            import io
            import zipfile

            client = get_client()
            items = []  # (image, caption)
            npy_data = []  # [(filename, numpy_array)]

            # Helper para consumir bytes .npy -> array (validación) -> item galería y colecta para ZIP
            def _consume_npy(bytes_buf: bytes, title: str, expect_first_band: bool = False, explicit_name: str | None = None):
                nonlocal items, npy_data
                if not bytes_buf:
                    return
                try:
                    arr = np.load(io.BytesIO(bytes_buf))
                except Exception:
                    return
                # Generar imagen para galería
                try:
                    if expect_first_band and arr.ndim == 3:
                        arr2d = arr[:, :, 0]
                    else:
                        arr2d = arr if arr.ndim == 2 else (arr[:, :, 0] if arr.ndim == 3 else arr)
                    items.append((normalize_to_uint8(arr2d), title))
                except Exception:
                    # Si falla visualización, igual empaquetamos el array crudo
                    pass
                # Guardar array para ZIP con un nombre seguro
                safe = (explicit_name or title).lower().replace(" ", "_") + ".npy"
                npy_data.append((safe, arr))

            # 1) Thermal
            try:
                r = client.render_thermal()
                if r.get("status") == "ok":
                    _consume_npy(r.get("npy_bytes"), "Thermal", expect_first_band=True, explicit_name="thermal")
            except Exception:
                pass

            # 2) Depth
            try:
                r = client.render_depth()
                if r.get("status") == "ok":
                    _consume_npy(r.get("npy_bytes"), "Depth", expect_first_band=False, explicit_name="depth")
            except Exception:
                pass

            # 3) Air renders & temperature map
            for endpoint, title, first_band in [
                ("/render/air/blackbody", "Blackbody Air", True),
                ("/render/air/transmittance", "Transmittance Air", True),
                ("/render/air/contribution", "Contribution Air", True),
                ("/render/temperature/map", "Temperature Map", False),
            ]:
                try:
                    r = client.session.get(f"{client.base_url}{endpoint}")
                    r.raise_for_status()
                    _consume_npy(r.content, title, expect_first_band=first_band)
                except Exception:
                    continue

            # Empaquetar ZIP completamente en memoria y luego volcar a un único archivo temporal
            zip_path = None
            try:
                mem_zip = io.BytesIO()
                with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, arr in npy_data:
                        try:
                            buf = io.BytesIO()
                            np.save(buf, arr)
                            zf.writestr(fname, buf.getvalue())
                        except Exception:
                            continue
                mem_zip.seek(0)

                # Compatibilidad: gr.File espera normalmente una ruta a archivo
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tf:
                    tf.write(mem_zip.getvalue())
                    zip_path = tf.name
            except Exception:
                zip_path = None

            return items, zip_path

        visualization_section["run_sim_btn"].click(
            fn=run_simulation_cb,
            inputs=[],
            outputs=[visualization_section["gallery"], visualization_section["download_zip"]],
        )

    # Carga de escena (upload / ruta servidor / escenas default miThermal)
        # Helper: construir updates de prefill para la pestaña Config a partir de la cámara actual
        def _prefill_updates_from_config(cfg: Dict | None):
            try:
                import math
                if not isinstance(cfg, dict):
                    return (gr.update(),) * 19
                
                spp = cfg.get("spp")
                k_val = None
                if isinstance(spp, (int, float)) and spp > 0:
                    try:
                        k_val = int(round(math.log2(float(spp))))
                    except Exception:
                        k_val = None
                
                wavelengths = cfg.get("wavelengths") or []
                wl_min = int(min(wavelengths)) if wavelengths else None
                wl_max = int(max(wavelengths)) if wavelengths else None
                bands = cfg.get("num_bands")

                return (
                    gr.update(value=k_val), gr.update(value=cfg.get("width")), gr.update(value=cfg.get("height")),
                    gr.update(value=cfg.get("rotate_x")), gr.update(value=cfg.get("rotate_y")), gr.update(value=cfg.get("rotate_z")),
                    gr.update(value=cfg.get("translate_x")), gr.update(value=cfg.get("translate_y")), gr.update(value=cfg.get("translate_z")),
                    gr.update(value=cfg.get("fov")),
                    gr.update(value=cfg.get("theta")), gr.update(value=cfg.get("phi")), gr.update(value=cfg.get("radius")),
                    gr.update(value=cfg.get("target_x")), gr.update(value=cfg.get("target_y")), gr.update(value=cfg.get("target_z")),
                    gr.update(value=wl_min), gr.update(value=wl_max), gr.update(value=bands),
                    gr.update(value=cfg),
                )
            except Exception as e:
                logger.error(f"_prefill_updates_from_config fallo: {e}")
                return (gr.update(),) * 19

        # Actualizar label del slider SPP cuando el usuario cambia k
        def _update_spp_label(k):
            try:
                kv = int(float(k)) if k is not None else None
                if kv is None:
                    return gr.update()
                val = int(2 ** kv)
                return gr.update(label=f"SPP = {val}")
            except Exception:
                return gr.update()

        # Enlazar cambio del slider para refrescar el label dinámico
        # Nota: devolvemos un update del propio slider
        # para no modificar su valor, solo su etiqueta.
        

        def upload_zip_and_prefill(load_type_value, zip_file):
            """Sube un ZIP (Escena o miTransfer) y retorna 22 outputs.

            Orden de outputs (22):
              1 image, 2 scene_info(str|update), 3 selector1(update), 4 scene_json(dict),
              5 selector2(update), 6 air_suggest_list(update), 7-21 (15 config updates), 22 air_plot(fig|None)
            """
            # Helpers
            EMPTY_SELECTOR = gr.update(choices=[], value=None)
            EMPTY_SUGGEST = gr.update(choices=[], value=None)
            PLACEHOLDER_CONFIG = [gr.update()]*15  # spp,width,height,rx,ry,rz,tx,ty,tz,fov,wl_min,wl_max,bands,air_temp,config_info

            def _error_tuple(msg: str):
                return (
                    None,                      # image
                    gr.update(value=msg),       # scene_info
                    EMPTY_SELECTOR,             # selector1
                    {},                         # scene_json
                    EMPTY_SELECTOR,             # selector2
                    EMPTY_SUGGEST,              # air_suggest_list
                    *PLACEHOLDER_CONFIG,        # 15 updates
                    None,                       # air_plot
                )

            if zip_file is None:
                return _error_tuple("❌ Sube un archivo .zip")
            path = getattr(zip_file, "name", None)
            if not path:
                return _error_tuple("❌ Archivo inválido")

            client = get_client()
            img_pil = None
            info_text = ""
            scene_json = {}
            labels1: List[str] = []
            labels2: List[str] = []
            try:
                if (load_type_value or '').lower().startswith('mit'):
                    # miTransfer
                    resp = client.post_mithermal(path)
                    try:
                        if isinstance(resp, dict) and resp.get('image_base64'):
                            import io
                            import base64
                            from PIL import Image
                            img_bytes = base64.b64decode(resp['image_base64'])
                            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    except Exception:
                        img_pil = None
                else:
                    img_pil2, info_text_scene, scene_json_scene, labelsA, labelsB = upload_zip_file(zip_file)
                    img_pil = img_pil2
                    info_text = info_text_scene
                    scene_json = scene_json_scene
                    labels1 = labelsA
                    labels2 = labelsB
                if not info_text:
                    info_text, scene_json, labels_built = _summarize_scene_and_update_state()
                    labels1 = labels_built
                    labels2 = labels_built
            except Exception as e:
                return _error_tuple(f"❌ Error subiendo: {e}")

            # Config prefill
            try:
                cfg = client.get_camera_config()
            except Exception:
                cfg = None
            updates = list(_prefill_updates_from_config(cfg))  # esperado 14 (incluye config_info al final)
            try:
                air_temp = client.get_air_temperature()
            except Exception:
                air_temp = None
            # Asegurar longitud y colocar temperatura antes de config_info
            if len(updates) == 14:
                updates.insert(-1, gr.update(value=air_temp))  # ahora 15
            else:
                while len(updates) < 14:
                    updates.append(gr.update())
                updates.insert(-1, gr.update(value=air_temp))
            # Sugerencias atenuación
            try:
                air_suggest_update = air_suggest_cb()
            except Exception:
                air_suggest_update = EMPTY_SUGGEST
            # Plot atenuación
            try:
                air_fig = air_plot_cb()
            except Exception:
                air_fig = None

            return (
                img_pil,
                info_text,
                gr.update(choices=labels1 or [], value=None),
                scene_json,
                gr.update(choices=labels2 or [], value=None),
                air_suggest_update,
                *updates,
                air_fig,
            )

        upload_section["upload_btn"].click(
            fn=upload_zip_and_prefill,
            inputs=[upload_section["load_type"], upload_section["zip_file"]],
            outputs=[
                upload_section["render_image"],
                upload_section["scene_info"],
                og_section["selector"],
                upload_section["scene_json"],
                og_section["selector"],
                    # Sugerencias de atenuación (precargadas)
                    config_section["air_suggest_list"],
                # Prefill de Config
                config_section["camera_spp"],
                config_section["camera_width"],
                config_section["camera_height"],
                config_section["rotate_x"],
                config_section["rotate_y"],
                config_section["rotate_z"],
                config_section["translate_x"],
                config_section["translate_y"],
                config_section["translate_z"],
                config_section["fov"],
                config_section["wl_min"],
                config_section["wl_max"],
                config_section["bands"],
                config_section["air_temperature"],
                config_section["config_info"],
                # Plot de atenuación actual
                config_section["air_plot"],
            ],
        )

        # --- Escenas predeterminadas (sugerir auto, seleccionar y refrescar) ---
        def default_suggest_cb():
            client = get_client()
            data = client.suggest_default_scenes()
            if isinstance(data, dict) and data.get("status") == "error":
                return gr.update(choices=[], value=None)
            try:
                choices = sorted([str(x) for x in data])
            except Exception:
                choices = []
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def select_default_cb(file_name: str):
            """Selecciona escena default y retorna 22 outputs consistentes."""
            client = get_client()

            EMPTY_SELECTOR = gr.update(choices=[], value=None)
            EMPTY_SUGGEST = gr.update(choices=[], value=None)
            PLACEHOLDER_CONFIG = [gr.update()]*15

            def _empty(msg: str):
                return (
                    None,                  # image
                    msg,                   # scene_info
                    EMPTY_SELECTOR,        # selector1
                    {},                    # scene_json
                    EMPTY_SELECTOR,        # selector2
                    EMPTY_SUGGEST,         # air_suggest_list
                    *PLACEHOLDER_CONFIG,   # 15 config updates
                    None,                  # air_plot
                )

            if not file_name:
                return _empty("❌ Seleccione una escena predeterminada")
            try:
                client.select_default_scene(file_name)
            except Exception:
                # Seguimos aunque falle selección
                pass
            # Actualizar estado objetos
            try:
                info_text, scene_json, labels = _summarize_scene_and_update_state()
            except Exception:
                info_text, scene_json, labels = "⚠️ No se pudo refrescar escena", {}, []
            img_pil = None  # No siempre hay render inmediato
            # Config
            try:
                cfg = client.get_camera_config()
            except Exception:
                cfg = None
            updates = list(_prefill_updates_from_config(cfg))
            try:
                air_temp = client.get_air_temperature()
            except Exception:
                air_temp = None
            if len(updates) == 14:
                updates.insert(-1, gr.update(value=air_temp))
            else:
                while len(updates) < 14:
                    updates.append(gr.update())
                updates.insert(-1, gr.update(value=air_temp))
            # Sugerencias
            try:
                air_suggest_update = air_suggest_cb()
            except Exception:
                air_suggest_update = EMPTY_SUGGEST
            # Plot
            try:
                air_fig = air_plot_cb()
            except Exception:
                air_fig = None
            return (
                img_pil,
                info_text,
                gr.update(choices=labels or [], value=None),
                scene_json,
                gr.update(choices=labels or [], value=None),
                air_suggest_update,
                *updates,
                air_fig,
            )

        # Auto-sugerir al cargar la interfaz (además de atenuación de aire)
        def bootstrap_default_suggest():
            return default_suggest_cb()

        # Asignar el valor del dropdown automáticamente al cargar
        interface.load(
            fn=bootstrap_default_suggest,
            inputs=[],
            outputs=[upload_section["default_suggest"]],
        )

        # Botón Recargar por defecto: repoblar dropdown de escenas predeterminadas
        if "reload_default_btn" in upload_section:
            upload_section["reload_default_btn"].click(
                fn=default_suggest_cb,
                inputs=[],
                outputs=[upload_section["default_suggest"]],
            )

        upload_section["select_default_btn"].click(
            fn=select_default_cb,
            inputs=[upload_section["default_suggest"]],
            outputs=[
                upload_section["render_image"],
                upload_section["scene_info"],
                og_section["selector"],
                upload_section["scene_json"],
                og_section["selector"],
                # Sugerencias de atenuación (precargadas)
                config_section["air_suggest_list"],
                # Prefill de Config
                config_section["camera_spp"],
                config_section["camera_width"],
                config_section["camera_height"],
                config_section["rotate_x"],
                config_section["rotate_y"],
                config_section["rotate_z"],
                config_section["translate_x"],
                config_section["translate_y"],
                config_section["translate_z"],
                config_section["fov"],
                config_section["wl_min"],
                config_section["wl_max"],
                config_section["bands"],
                config_section["air_temperature"],
                config_section["config_info"],
                # Plot de atenuación actual
                config_section["air_plot"],
            ],
        )

        # (Eliminados) callbacks específicos de miThermal (post/get) tras unificación de carga.
        if upload_section["load_btn"] and upload_section["server_path"]:
            def load_server_and_prefill(scene_path: str):
                out_img, info_text, scene_json, labels1, labels2 = load_server_scene(scene_path)
                cfg = None
                try:
                    cfg = get_client().get_camera_config()
                except Exception:
                    cfg = None
                updates = list(_prefill_updates_from_config(cfg))
                # Temperatura del aire
                try:
                    air_temp = get_client().get_air_temperature()
                except Exception:
                    air_temp = None
                updates.insert(-1, gr.update(value=air_temp))
                # Precargar sugerencias de atenuación
                try:
                    air_suggest_update = air_suggest_cb()
                except Exception:
                    air_suggest_update = gr.update(choices=[], value=None)
                # Plot actual de atenuación
                try:
                    air_fig = air_plot_cb()
                except Exception:
                    air_fig = None
                return (
                    out_img,
                    info_text,
                    gr.update(choices=labels1 or [], value=None),
                    scene_json,
                    gr.update(choices=labels2 or [], value=None),
                    air_suggest_update,
                    *updates,
                    air_fig,
                )

            upload_section["load_btn"].click(
                fn=load_server_and_prefill,
                inputs=[upload_section["server_path"]],
                outputs=[
                    upload_section["render_image"],
                    upload_section["scene_info"],
                    og_section["selector"],
                    upload_section["scene_json"],
                    og_section["selector"],
                    # Sugerencias de atenuación (precargadas)
                    config_section["air_suggest_list"],
                    # Prefill de Config
                    config_section["camera_spp"],
                    config_section["camera_width"],
                    config_section["camera_height"],
                    config_section["rotate_x"],
                    config_section["rotate_y"],
                    config_section["rotate_z"],
                    config_section["translate_x"],
                    config_section["translate_y"],
                    config_section["translate_z"],
                    config_section["fov"],
                    config_section["wl_min"],
                    config_section["wl_max"],
                    config_section["bands"],
                    config_section["air_temperature"],
                    config_section["config_info"],
                    # Plot de atenuación actual
                    config_section["air_plot"],
                ],
            )

        # --- Lógica unificada Objeto/Instancia/Familia ---
        def og_refresh_selector(mode: str):
            mode_l = (mode or "").lower()
            if mode_l.startswith("obj"):
                choices = sorted(viewer_state.object_id_mapping.keys())
            elif mode_l.startswith("inst"):
                choices = sorted(viewer_state.object_groups_instance.keys())
            else:
                choices = sorted(viewer_state.object_groups_family.keys())
            return (
                gr.update(choices=choices, value=None),
                gr.update(choices=[], value=[]),
                gr.update(value="Seleccione una opción"),
            )

        og_section["mode_radio"].change(
            fn=og_refresh_selector,
            inputs=[og_section["mode_radio"]],
            outputs=[
                og_section["selector"], 
                og_section["members"], 
                og_section["info_text"]
                ],
        )

        def og_selector_changed(mode: str, selection: str):
            mode_l = (mode or "").lower()
            if not selection:
                return (
                    gr.update(choices=[], value=[]),
                    gr.update(value="Seleccione una opción"),
                    None,
                    None,
                )
            if mode_l.startswith("obj"):
                mv, info, _of, _t, plot = view_object_3d(selection)
                
                # Agregar información de familia si pertenece a una
                oid = viewer_state.object_id_mapping.get(selection)
                if oid:
                    try:
                        client = get_client()
                        preview = client.session.get(
                            f"{client.base_url}/object/families/preview/{oid}"
                        ).json()
                        
                        if preview.get("has_family"):
                            family_info = (
                                f"\n\n👨‍👩‍👧‍👦 **Familia:** {preview['family_name']}\n"
                                f"**Miembros:** {preview['count']} objetos\n"
                                f"💡 *Cambia a modo 'Familia' para actualizar "
                                f"todos a la vez*"
                            )
                            info = info + family_info if info else family_info
                    except Exception as e:
                        logger.debug(f"Error obteniendo info de familia: {e}")
                
                return (
                    gr.update(choices=[selection], value=[selection]),
                    gr.update(value=info),
                    mv,
                    plot,
                )
            elif mode_l.startswith("inst"):
                oids = viewer_state.object_groups_instance.get(selection, [])
            else:
                oids = viewer_state.object_groups_family.get(selection, [])
            labels = [
                label for label, oid in viewer_state.object_id_mapping.items()
                if oid in oids
            ]
            labels_sorted = sorted(labels)
            if labels_sorted:
                mv, info, _of, _t, plot = view_object_3d(labels_sorted[0])
                return (
                    gr.update(choices=labels_sorted, value=[]),
                    gr.update(value=info),
                    mv,
                    plot
                )
            return (
                gr.update(choices=[], value=[]),
                gr.update(value=f"Sin miembros para {selection}"),
                None,
                None
            )

        og_section["selector"].change(
            fn=og_selector_changed,
            inputs=[og_section["mode_radio"], og_section["selector"]],
            outputs=[
                og_section["members"],
                og_section["info_text"],
                og_section["model_viewer"],
                og_section["emissivity_plot"],
            ],
        )

        def og_apply_update(mode: str, selection: str, temp_value, emissivity_file):
            """
            Actualiza objeto(s) según el modo seleccionado.
            
            - Objeto: Actualiza solo el objeto seleccionado
            - Familia: Actualiza todos los miembros de la familia
            
            Args:
                mode: "Objeto", "Instancia", o "Familia"
                selection: Objeto/grupo seleccionado
                temp_value: Nueva temperatura (opcional)
                emissivity_file: Archivo de emisividad (opcional)
            """
            mode_l = (mode or '').lower()
            client = get_client()
            
            # Validar que hay algo que actualizar
            if not selection:
                return gr.update(value='❌ Debe seleccionar un objeto')
            
            # Obtener el object_id
            if mode_l.startswith('obj'):
                oid = viewer_state.object_id_mapping.get(selection)
                if not oid:
                    return gr.update(value='❌ Objeto no encontrado')
            elif mode_l.startswith('inst'):
                # Para instancia, usar el antiguo método
                targets = viewer_state.object_groups_instance.get(selection, [])
                if not targets:
                    return gr.update(value='❌ No hay instancias seleccionadas')
                # Fallback al método antiguo para instancias
                path = getattr(emissivity_file, 'name', None)
                if not path:
                    return gr.update(value='❌ Suba archivo de emisividad')
                try:
                    user_temp = (
                        float(temp_value) if temp_value not in (None, '') else None
                    )
                except Exception:
                    user_temp = None
                
                objs_payload = []
                for target_oid in targets:
                    if user_temp is not None:
                        t = user_temp
                    else:
                        try:
                            obj_resp = client.get_object(target_oid)
                            if obj_resp.get('status') == 'success':
                                odata = obj_resp.get('object', {})
                            else:
                                odata = obj_resp.get('object', obj_resp)
                            t = float(odata.get('temperature', 300.0))
                        except Exception:
                            t = 300.0
                    objs_payload.append({'id': target_oid, 'temperature': t})
                
                res = client.update_objects_with_emissivity(objs_payload, path)
                if res.get('status') == 'error':
                    return gr.update(value=f"❌ Error: {res.get('detail')}")
                msg = res.get('message') if isinstance(res, dict) else 'OK'
                return gr.update(
                    value=f"✅ Actualizados {len(objs_payload)} objeto(s). {msg}"
                )
            else:
                # Modo Familia
                targets = viewer_state.object_groups_family.get(selection, [])
                if not targets:
                    return gr.update(value='❌ No hay objetos en esta familia')
                oid = targets[0] if targets else None
                if not oid:
                    return gr.update(value='❌ Familia vacía')
            
            # Preparar parámetros para el nuevo endpoint
            params = {"mode": "Familia" if mode_l.startswith("fam") else "Objeto"}
            
            if temp_value not in (None, ''):
                try:
                    params["temperature"] = float(temp_value)
                except ValueError:
                    return gr.update(value='❌ Temperatura inválida')
            
            # Manejar archivo de emisividad (TODO: subir primero al backend)
            emissivity_file_path = getattr(emissivity_file, 'name', None)
            if emissivity_file_path:
                # Por ahora, usar el nombre del archivo
                # En una versión futura, se debería subir el archivo primero
                params["emissivity_file"] = emissivity_file_path
            
            # Validar que hay al menos una propiedad para actualizar
            if "temperature" not in params and "emissivity_file" not in params:
                return gr.update(
                    value='❌ Debe especificar temperatura o archivo de emisividad'
                )
            
            try:
                # Llamar al nuevo endpoint
                params["object_id"] = oid
                response = client.session.put(
                    f"{client.base_url}/object/update-with-mode",
                    params=params
                )
                response.raise_for_status()
                result = response.json()
                
                # Construir mensaje de éxito
                count = result.get("count", 0)
                mode_text = result.get("mode", "")
                
                if mode_text == "family":
                    family_name = result.get("family_name", "")
                    msg = (
                        f"✅ Familia '{family_name}' actualizada\n"
                        f"📦 {count} objetos modificados"
                    )
                else:
                    msg = f"✅ Objeto '{oid}' actualizado"
                
                props_updated = result.get("properties_updated", [])
                if props_updated:
                    msg += f"\n🔧 Propiedades: {', '.join(props_updated)}"
                
                return gr.update(value=msg)
                
            except Exception as e:
                logger.error(f"Error en og_apply_update: {e}")
                return gr.update(value=f"❌ Error: {str(e)}")

        og_section['apply_update_btn'].click(
            fn=og_apply_update,
            inputs=[og_section['mode_radio'], og_section['selector'], og_section['temp_input'], og_section['emissivity_file']],
            outputs=[og_section['info_text']],
        )

        # Batch update con atenuación eliminado según requerimiento del usuario.

        def preview_emissivity_file(emissivity_file):
            if emissivity_file is None:
                return None
            try:
                path = getattr(emissivity_file, "name", None)
                if not path:
                    return None
                import pathlib

                text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
                wavelengths: List[float] = []
                reflection: List[float] = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) < 2:
                        continue
                    try:
                        wl = float(parts[0])
                        ref = float(parts[1])
                        wavelengths.append(wl)
                        reflection.append(ref)
                    except ValueError:
                        continue
                if not wavelengths or len(wavelengths) != len(reflection):
                    return None
                max_ref = max(reflection)
                if max_ref > 1.5:
                    emissivity = [1 - r / 100.0 for r in reflection]
                else:
                    emissivity = [1 - r for r in reflection]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=wavelengths,
                    y=emissivity,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=emissivity,
                        colorscale='Magma',
                        showscale=True,
                        colorbar=dict(title="Emisividad")
                    ),
                    hovertemplate='<b>λ:</b> %{x:.1f}<br><b>Emisividad:</b> %{y:.4f}<extra></extra>'
                ))
                fig.update_layout(
                    title='Previsualización Emisividad',
                    xaxis_title='Longitud de onda (nm)',
                    yaxis_title='Emisividad (0-1)',
                    hovermode='closest',
                    template='plotly_white',
                    height=350,
                    yaxis=dict(range=[0, 1.05])
                )
                return fig
            except Exception as e:
                logger.warning(f"preview_emissivity_file fallo: {e}")
                return None
            return None

        og_section["emissivity_file"].change(
            fn=preview_emissivity_file,
            inputs=[og_section["emissivity_file"]],
            outputs=[og_section["emissivity_plot"]],
        )

        # Config unificada: cámara + espectro + aire
        def apply_all_config_cb(
            spp_k, width, height, wl_min, wl_max, bands, air_temperature, air_file,
            rx, ry, rz, tx, ty, tz, fov,
            theta, phi, radius, target_x, target_y, target_z
        ):
            client = get_client()
            results: Dict[str, Dict] = {}
            # Cámara
            try:
                k = int(float(spp_k)) if spp_k is not None else None
                spp_val = 2 ** k if k is not None else None
                
                # Preparamos argumentos para update_camera_config
                cam_args = {
                    "spp": int(spp_val) if spp_val is not None else None,
                    "width": int(width) if width is not None else None,
                    "height": int(height) if height is not None else None,
                    "fov": float(fov) if fov is not None else None,
                }
                
                # Si hay valores esféricos, los priorizamos (el backend se encarga de la lógica)
                if all(v is not None for v in [theta, phi, radius]):
                    cam_args.update({
                        "theta": float(theta),
                        "phi": float(phi),
                        "radius": float(radius),
                        "target_x": float(target_x) if target_x is not None else 0.0,
                        "target_y": float(target_y) if target_y is not None else 0.0,
                        "target_z": float(target_z) if target_z is not None else 0.0,
                    })
                else:
                    # Si no, usamos cartesianas
                    cam_args.update({
                        "rotate_x": float(rx) if rx is not None else None,
                        "rotate_y": float(ry) if ry is not None else None,
                        "rotate_z": float(rz) if rz is not None else None,
                        "translate_x": float(tx) if tx is not None else None,
                        "translate_y": float(ty) if ty is not None else None,
                        "translate_z": float(tz) if tz is not None else None,
                    })
                
                results["camera"] = client.update_camera_config(**cam_args)
            except Exception as e:
                results["camera"] = {"status": "error", "detail": str(e)}
            
            # Espectro
            try:
                wl_min_i = int(wl_min) if wl_min is not None else None
                wl_max_i = int(wl_max) if wl_max is not None else None
                bands_i = int(bands) if bands is not None else None
                if all(v is not None for v in [wl_min_i, wl_max_i, bands_i]):
                    results["spectrum"] = client.update_wavelengths(wl_min_i, wl_max_i, bands_i)
            except Exception as e:
                results["spectrum"] = {"status": "error", "detail": str(e)}
            
            # Aire (resto igual...)
            air_plot = None
            try:
                if air_temperature is not None and air_temperature != "":
                    try:
                        t = float(air_temperature)
                        if t > 0:
                            results["air_temperature"] = client.set_air_temperature(t)
                    except Exception: pass
                
                if air_file is not None:
                    path = getattr(air_file, "name", None)
                    if path:
                        results["air"] = client.upload_air_attenuation(path)
                
                text = client.get_air_attenuation()
                try:
                    xs, ys = parse_air_text_to_xy(text)
                    if xs and ys:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Atenuación'))
                        fig.update_layout(title='Atenuación del Aire', height=350)
                        air_plot = fig
                except Exception: pass
            except Exception as e:
                results["air"] = {"status": "error", "detail": str(e)}
                
            status_text = "\n".join([f"{k}: {v.get('status', 'ok') if isinstance(v, dict) else 'ok'}" for k, v in results.items()])
            return results, status_text, air_plot

        config_section["apply_all_btn"].click(
            fn=apply_all_config_cb,
            inputs=[
                config_section["camera_spp"], config_section["camera_width"], config_section["camera_height"],
                config_section["wl_min"], config_section["wl_max"], config_section["bands"],
                config_section["air_temperature"], config_section["air_file"],
                config_section["rotate_x"], config_section["rotate_y"], config_section["rotate_z"],
                config_section["translate_x"], config_section["translate_y"], config_section["translate_z"],
                config_section["fov"],
                config_section["theta"], config_section["phi"], config_section["radius"],
                config_section["target_x"], config_section["target_y"], config_section["target_z"],
            ],
            outputs=[
                config_section["config_info"], config_section["config_status"], config_section["air_plot"],
            ],
        )

        def export_cam_cb():
            import json, tempfile
            res = get_client().export_camera_spatial_config()
            if res.get("status") == "success":
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as f:
                    json.dump(res["data"], f)
                    return f.name
            return None

        config_section["export_cam_btn"].click(fn=export_cam_cb, outputs=[gr.File(label="Cam Config JSON")])

        def import_cam_cb(file):
            import json
            if file is None: return "No file selected"
            try:
                with open(file.name, "r") as f:
                    data = json.load(f)
                res = get_client().import_camera_spatial_config(data)
                return f"Import status: {res.get('status')}"
            except Exception as e:
                return f"Error: {str(e)}"

        config_section["import_cam_btn"].click(fn=import_cam_cb, inputs=[gr.File()], outputs=[config_section["config_status"]])


        # Utilidad local: parsear texto de aire a pares (x, y)
        def parse_air_text_to_xy(text: str):
            xs: List[float] = []
            ys: List[float] = []
            for line in (text or "").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace("\t", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except Exception:
                    continue
            return xs, ys

        # Nuevo: botón "Cargar Config" manual
        def load_config_cb():
            cfg = None
            try:
                cfg = get_client().get_camera_config()
            except Exception:
                cfg = None
            updates = list(_prefill_updates_from_config(cfg))
            # Temperatura del aire
            try:
                air_temp = get_client().get_air_temperature()
            except Exception:
                air_temp = None
            updates.insert(-1, gr.update(value=air_temp))
            return tuple(updates)

        config_section["load_config_btn"].click(
            fn=load_config_cb,
            inputs=[],
            outputs=[
                config_section["camera_spp"],
                config_section["camera_width"],
                config_section["camera_height"],
                config_section["rotate_x"],
                config_section["rotate_y"],
                config_section["rotate_z"],
                config_section["translate_x"],
                config_section["translate_y"],
                config_section["translate_z"],
                config_section["fov"],
                config_section["theta"],
                config_section["phi"],
                config_section["radius"],
                config_section["target_x"],
                config_section["target_y"],
                config_section["target_z"],
                config_section["wl_min"],
                config_section["wl_max"],
                config_section["bands"],
                config_section["air_temperature"],
                config_section["config_info"],
            ],
        )

        # Cambiar etiqueta del slider dinámicamente cuando el usuario ajusta k
        config_section["camera_spp"].change(
            fn=_update_spp_label,
            inputs=[config_section["camera_spp"]],
            outputs=[config_section["camera_spp"]],
        )

        # Callback para aplicar solo configuración espectral
        def apply_spectrum_config_cb(wl_min, wl_max, bands):
            """Actualiza solo los parámetros espectrales."""
            try:
                client = get_client()
                result = client.update_camera_config(
                    wavelength_min=int(wl_min),
                    wavelength_max=int(wl_max),
                    num_bands=int(bands)
                )
                if result.get("status") == "success":
                    msg = f"✅ Configuración espectral actualizada:\n• λ: {wl_min}-{wl_max} nm\n• Bandas: {bands}"
                else:
                    msg = f"❌ Error: {result.get('detail', 'Error desconocido')}"
                return msg
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        config_section["spectrum_apply_btn"].click(
            fn=apply_spectrum_config_cb,
            inputs=[
                config_section["wl_min"],
                config_section["wl_max"],
                config_section["bands"],
            ],
            outputs=[config_section["spectrum_status"]],
        )

        

        # Nuevo: sugerir y aplicar atenuación del aire
        def air_suggest_cb():
            client = get_client()
            data = client.suggest_air_attenuation()
            if isinstance(data, dict) and data.get("status") == "error":
                return gr.update(choices=[], value=None)
            try:
                choices = sorted([str(x) for x in data])
            except Exception:
                choices = []
            return gr.update(choices=choices, value=choices[0] if choices else None)

        # config_section["air_suggest_btn"].click(
        #     fn=air_suggest_cb,
        #     inputs=[],
        #     outputs=[config_section["air_suggest_list"]],
        # )

        def air_apply_suggest_cb(file_name: str):
            client = get_client()
            res = client.set_air_attenuation_by_filename(file_name) if file_name else {"status": "error"}
            # refrescar plot
            air_plot = None
            try:
                text = client.get_air_attenuation()
                xs, ys = parse_air_text_to_xy(text)
                if xs and ys:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines',
                        name='Atenuación',
                        line=dict(color='#D55E00', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Atenuación:</b> %{y:.4f}<extra></extra>'
                    ))
                    fig.update_layout(
                        title='Atenuación del Aire',
                        xaxis_title='Longitud de Onda (nm)',
                        yaxis_title='Atenuación',
                        hovermode='closest',
                        template='plotly_white',
                        height=350
                    )
                    air_plot = fig
            except Exception:
                pass
            status = res.get("status", "success") if isinstance(res, dict) else "success"
            msg = f"Atenuación: {status}"
            return msg, air_plot

        config_section["air_apply_suggest_btn"].click(
            fn=air_apply_suggest_cb,
            inputs=[config_section["air_suggest_list"]],
            outputs=[config_section["config_status"], config_section["air_plot"]],
        )

        # Helper: construir plot de atenuación actual
        def air_plot_cb():
            try:
                client = get_client()
                text = client.get_air_attenuation()
                xs, ys = parse_air_text_to_xy(text)
                if xs and ys:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines',
                        name='Atenuación',
                        line=dict(color='#D55E00', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Atenuación:</b> %{y:.4f}<extra></extra>'
                    ))
                    fig.update_layout(
                        title='Atenuación del Aire',
                        xaxis_title='Longitud de Onda (nm)',
                        yaxis_title='Atenuación',
                        hovermode='closest',
                        template='plotly_white',
                        height=350
                    )
                    return fig
            except Exception:
                return None
            return None

        # Precargar sugerencias y plot de atenuación al cargar la interfaz
        def air_bootstrap_cb():
            dd = air_suggest_cb()
            fig = air_plot_cb()
            # temperatura del aire
            try:
                air_temp = get_client().get_air_temperature()
            except Exception:
                air_temp = None
            return dd, fig, gr.update(value=air_temp)

        interface.load(
            fn=air_bootstrap_cb,
            inputs=[],
            outputs=[config_section["air_suggest_list"], config_section["air_plot"], config_section["air_temperature"]],
        )

        # --------------------------------------------------------------------------------------
        # Callback para interpolación de cámara
        # --------------------------------------------------------------------------------------
        def camera_interpolation_cb(
            mode, ox, oy, oz, ex, ey, ez, st, sa, sr, et, ea, er, lock_a, tx, ty, tz, steps
        ):
            client = get_client()
            try:
                if mode == "Lineal":
                    res = client.session.post(f"{client.base_url}/scene/camera/interpolation", json={
                        "origin": [ox, oy, oz], "end": [ex, ey, ez],
                        "tracked_point": [tx, ty, tz], "num_steps": int(steps)
                    })
                else:
                    res = client.generate_spherical_interpolation(
                        start_theta=st, end_theta=et, start_azimuth=sa, end_azimuth=ea,
                        start_radius=sr, end_radius=er, lock_azimuth_to_end=lock_a,
                        tracked_point=[tx, ty, tz], num_steps=int(steps)
                    )
                    if isinstance(res, dict) and res.get("status") == "success":
                        frames = res["data"]
                        return f"✅ Generados {len(frames)} frames (Esférico)", frames
                    else:
                        return f"❌ Error: {res}", []

                res.raise_for_status()
                frames = res.json()
                return f"✅ Generados {len(frames)} frames ({mode})", frames
            except Exception as e:
                return f"❌ Error: {str(e)}", []

        def render_camera_animation_cb(mode, ox, oy, oz, ex, ey, ez, st, sa, sr, et, ea, er, lock_a, tx, ty, tz, steps):
            client = get_client()
            try:
                if mode == "Lineal":
                    payload = {"origin": [ox, oy, oz], "end": [ex, ey, ez], "tracked_point": [tx, ty, tz], "num_steps": int(steps)}
                    url = f"{client.base_url}/scene/camera/animation/render"
                else:
                    payload = {
                        "start_theta": st, "end_theta": et, "start_azimuth": sa, "end_azimuth": ea,
                        "start_radius": sr, "end_radius": er, "lock_azimuth_to_end": lock_a,
                        "tracked_point": [tx, ty, tz], "num_steps": int(steps)
                    }
                    url = f"{client.base_url}/scene/camera/animation/render/spherical"

                res = client.session.post(url, json=payload, timeout=600)
                res.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tf:
                    tf.write(res.content)
                    return f"✅ Animación renderizada ({mode})", tf.name
            except Exception as e:
                return f"❌ Error: {str(e)}", None

        def preview_camera_path_cb(mode, ox, oy, oz, ex, ey, ez, st, sa, sr, et, ea, er, lock_a, tx, ty, tz, steps):
            client = get_client()
            try:
                if mode == "Lineal":
                    payload = {"origin": [ox, oy, oz], "end": [ex, ey, ez], "tracked_point": [tx, ty, tz], "num_steps": int(steps)}
                    url = f"{client.base_url}/scene/camera/animation/preview"
                else:
                    payload = {
                        "start_theta": st, "end_theta": et, "start_azimuth": sa, "end_azimuth": ea,
                        "start_radius": sr, "end_radius": er, "lock_azimuth_to_end": lock_a,
                        "tracked_point": [tx, ty, tz], "num_steps": int(steps)
                    }
                    url = f"{client.base_url}/scene/camera/animation/preview/spherical"
                
                res = client.session.post(url, json=payload, timeout=300)
                res.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tf:
                    tf.write(res.content)
                    return f"✅ Preview generado ({mode})", tf.name
            except Exception as e:
                return f"❌ Error: {str(e)}", None

        def export_anim_cb(mode, ox, oy, oz, ex, ey, ez, st, sa, sr, et, ea, er, lock_a, tx, ty, tz, steps):
            import json
            client = get_client()
            if mode == "Lineal":
                data = {"origin": [ox, oy, oz], "end": [ex, ey, ez], "tracked_point": [tx, ty, tz], "num_steps": int(steps)}
            else:
                data = {
                    "start_theta": st, "end_theta": et, "start_azimuth": sa, "end_azimuth": ea,
                    "start_radius": sr, "end_radius": er, "lock_azimuth_to_end": lock_a,
                    "tracked_point": [tx, ty, tz], "num_steps": int(steps)
                }
            res = client.export_camera_animation(mode.lower(), data)
            if res.get("status") == "success":
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as f:
                    json.dump(res["data"], f)
                    return f.name
            return None

        def import_anim_cb(file):
            import json
            if file is None: return "No file selected", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            try:
                with open(file.name, "r") as f:
                    data = json.load(f)
                mode = data.get("mode", "linear")
                mode_label = "Lineal" if mode == "linear" else "Esférico"
                
                updates = [gr.update(value=mode_label)] # interp_mode
                
                if mode == "linear" and "linear_data" in data:
                    ld = data["linear_data"]
                    updates.extend([
                        gr.update(value=ld["origin"][0]), gr.update(value=ld["origin"][1]), gr.update(value=ld["origin"][2]),
                        gr.update(value=ld["end"][0]), gr.update(value=ld["end"][1]), gr.update(value=ld["end"][2]),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(value=ld["tracked_point"][0]), gr.update(value=ld["tracked_point"][1]), gr.update(value=ld["tracked_point"][2]),
                        gr.update(value=ld["num_steps"])
                    ])
                elif mode == "spherical" and "spherical_data" in data:
                    sd = data["spherical_data"]
                    updates.extend([
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(value=sd["start_theta"]), gr.update(value=sd["start_azimuth"]), gr.update(value=sd.get("start_radius", sd.get("radius"))),
                        gr.update(value=sd["end_theta"]), gr.update(value=sd["end_azimuth"]), gr.update(value=sd.get("end_radius", sd.get("radius"))),
                        gr.update(value=sd.get("lock_azimuth_to_end", False)),
                        gr.update(value=sd["tracked_point"][0]), gr.update(value=sd["tracked_point"][1]), gr.update(value=sd["tracked_point"][2]),
                        gr.update(value=sd["num_steps"])
                    ])
                else:
                    return "Invalid JSON format", *([gr.update()] * 17)
                
                return f"✅ Animación cargada ({mode_label})", *updates
            except Exception as e:
                return f"❌ Error: {str(e)}", *([gr.update()] * 17)

        interp_inputs = [
            camera_interp_section["interp_mode"],
            camera_interp_section["origin_x"], camera_interp_section["origin_y"], camera_interp_section["origin_z"],
            camera_interp_section["end_x"], camera_interp_section["end_y"], camera_interp_section["end_z"],
            camera_interp_section["start_theta"], camera_interp_section["start_azimuth"], camera_interp_section["start_radius"],
            camera_interp_section["end_theta"], camera_interp_section["end_azimuth"], camera_interp_section["end_radius"],
            camera_interp_section["lock_azimuth"],
            camera_interp_section["target_x"], camera_interp_section["target_y"], camera_interp_section["target_z"],
            camera_interp_section["num_steps"],
        ]

        camera_interp_section["generate_btn"].click(fn=camera_interpolation_cb, inputs=interp_inputs, outputs=[camera_interp_section["status_output"], camera_interp_section["interpolation_result"]])
        camera_interp_section["render_btn"].click(fn=render_camera_animation_cb, inputs=interp_inputs, outputs=[camera_interp_section["status_output"], camera_interp_section["animation_zip"]])
        camera_interp_section["preview_btn"].click(fn=preview_camera_path_cb, inputs=interp_inputs, outputs=[camera_interp_section["status_output"], camera_interp_section["preview_gif"]])
        camera_interp_section["export_anim_btn"].click(fn=export_anim_cb, inputs=interp_inputs, outputs=[gr.File(label="Anim Config JSON")])
        
        # Para import, necesitamos que devuelva updates a todos los campos
        interp_outputs = [camera_interp_section["status_output"]] + interp_inputs
        camera_interp_section["import_anim_btn"].click(fn=import_anim_cb, inputs=[gr.File()], outputs=interp_outputs)

        # Cache Management: conectar callbacks
        cache_section["refresh_stats_btn"].click(
            fn=refresh_cache_stats_cb,
            inputs=[],
            outputs=[
                cache_section["cache_stats_display"],
                cache_section["cache_status_output"],
            ],
        )

        cache_section["clear_cache_btn"].click(
            fn=clear_cache_cb,
            inputs=[],
            outputs=[
                cache_section["cache_stats_display"],
                cache_section["cache_status_output"],
            ],
        )

        # ============================================================================
        # Spectral Data Plotting: conectar callbacks
        # ============================================================================
        
        # Callback para mostrar/ocultar componentes según tipo de gráfico
        def update_spectral_ui(plot_type):
            """Actualiza qué controles están visibles según el tipo de gráfico."""
            show_object = plot_type in ["Emisividad", "Reflectancia"]
            show_temp = plot_type == "Radiancia Térmica"
            show_gas = plot_type == "Atenuación Atmosférica"
            show_wl = True  # Siempre mostrar rangos de longitud de onda
            
            return (
                gr.update(visible=show_object),  # object_select
                gr.update(visible=show_temp),     # temp_k
                gr.update(visible=show_gas),      # gas_type
                gr.update(visible=show_wl),       # wl_min
                gr.update(visible=show_wl),       # wl_max
            )
        
        spectral_section["plot_type"].change(
            fn=update_spectral_ui,
            inputs=[spectral_section["plot_type"]],
            outputs=[
                spectral_section["object_select"],
                spectral_section["temp_k"],
                spectral_section["gas_type"],
                spectral_section["wl_min"],
                spectral_section["wl_max"],
            ],
        )
        
        # Callback para generar gráfico
        def generate_spectral_plot_cb(plot_type, object_id, temp_k, gas_type, wl_min, wl_max):
            """Genera gráfico espectral según parámetros seleccionados."""
            try:
                client = get_client()
                fig = go.Figure()
                plot_info = ""
                title_text = "Datos Espectrales"
                ylabel_text = "Valor"
                yaxis_type = "linear"  # Por defecto lineal, puede ser "log" para logarítmico

                def clean_xy(x_values, y_values):
                    clean_x = []
                    clean_y = []
                    for x_val, y_val in zip(x_values or [], y_values or []):
                        if x_val is None or y_val is None:
                            continue
                        clean_x.append(float(x_val))
                        clean_y.append(float(y_val))
                    return clean_x, clean_y
                
                if plot_type == "Emisividad":
                    if not object_id:
                        return None, "❌ Por favor selecciona un objeto"
                    result = client.get_emissivity_spectrum(
                        object_id,
                        wavelength_min_nm=int(wl_min) if wl_min else None,
                        wavelength_max_nm=int(wl_max) if wl_max else None
                    )
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    values = data.get("values", [])
                    wavelengths, values = clean_xy(wavelengths, values)
                    if not values:
                        return None, "❌ No hay datos válidos de emisividad"
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths, y=values,
                        mode='lines+markers',
                        name='Emisividad',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Emisividad:</b> %{y:.4f}<extra></extra>'
                    ))
                    plot_info = f"✅ Emisividad de {object_id}\nRango: {min(values):.3f} - {max(values):.3f}"
                    title_text = f"Emisividad - {object_id}"
                    ylabel_text = "Emisividad (0-1)"
                    
                elif plot_type == "Reflectancia":
                    if not object_id:
                        return None, "❌ Por favor selecciona un objeto"
                    result = client.get_reflectance_spectrum(
                        object_id,
                        wavelength_min_nm=int(wl_min) if wl_min else None,
                        wavelength_max_nm=int(wl_max) if wl_max else None
                    )
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    values = data.get("values", [])
                    wavelengths, values = clean_xy(wavelengths, values)
                    if not values:
                        return None, "❌ No hay datos válidos de reflectancia"
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths, y=values,
                        mode='lines+markers',
                        name='Reflectancia',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Reflectancia:</b> %{y:.4f}<extra></extra>'
                    ))
                    plot_info = f"✅ Reflectancia de {object_id}\nRango: {min(values):.3f} - {max(values):.3f}"
                    title_text = f"Reflectancia - {object_id}"
                    ylabel_text = "Reflectancia (0-1)"
                    
                elif plot_type == "Radiancia Térmica":
                    result = client.get_blackbody_spectrum(
                        temperature_k=temp_k,
                        wavelength_min_nm=wl_min,
                        wavelength_max_nm=wl_max,
                        num_points=100
                    )
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    radiance = data.get("radiance", [])
                    wavelengths, radiance = clean_xy(wavelengths, radiance)
                    if not radiance:
                        return None, "❌ No hay datos válidos de radiancia"
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths, y=radiance,
                        mode='lines',
                        name=f'Cuerpo Negro {temp_k}K',
                        line=dict(color='orange', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Radiancia:</b> %{y:.3e} W/(m^3·sr)<extra></extra>'
                    ))
                    max_radiance = max(radiance) if radiance else 0
                    plot_info = f"✅ Cuerpo Negro a {temp_k}K\nRadiancia máxima: {max_radiance:.3e} W/(m³·sr)"
                    title_text = f"Radiancia Térmica - {temp_k}K"
                    ylabel_text = "Radiancia W/(m³·sr)"
                    
                elif plot_type == "Atenuación Atmosférica":
                    result = client.get_atmospheric_spectrum(
                        gas=gas_type,
                        wavelength_min_nm=int(wl_min) if wl_min else None,
                        wavelength_max_nm=int(wl_max) if wl_max else None
                    )
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    attenuation = data.get("attenuation", [])
                    transmittance = data.get("transmittance", [])

                    wavelengths_a, attenuation = clean_xy(wavelengths, attenuation)
                    wavelengths_t, transmittance = clean_xy(wavelengths, transmittance)
                    if not attenuation and not transmittance:
                        return None, "❌ No hay datos válidos de atenuación atmosférica"
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths_a, y=attenuation,
                        mode='lines',
                        name='Atenuación',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Atenuación:</b> %{y:.4f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths_t, y=transmittance,
                        mode='lines',
                        name='Transmitancia',
                        line=dict(color='green', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Transmitancia:</b> %{y:.4f}<extra></extra>'
                    ))
                    
                    attenuation_mean = np.mean(attenuation) if attenuation else 0.0
                    plot_info = (
                        f"✅ Atmósfera: {gas_type}\n"
                        f"Promedio Atenuación: {attenuation_mean:.3f}"
                    )
                    title_text = f"Atenuación Atmosférica - {gas_type}"
                    ylabel_text = "Atenuación / Transmitancia (escala log)"
                    yaxis_type = "log"  # Escala logarítmica para atenuación atmosférica
                
                # Configurar diseño del gráfico
                fig.update_layout(
                    title=title_text,
                    xaxis_title="Longitud de Onda (nm)",
                    yaxis_title=ylabel_text,
                    yaxis_type=yaxis_type,
                    hovermode='closest',
                    template='plotly_white',
                    height=500,
                    showlegend=True,
                )
                
                return fig, plot_info
                
            except Exception as e:
                error_msg = f"❌ Error generando gráfico: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return None, error_msg
        
        spectral_section["generate_plot_btn"].click(
            fn=generate_spectral_plot_cb,
            inputs=[
                spectral_section["plot_type"],
                spectral_section["object_select"],
                spectral_section["temp_k"],
                spectral_section["gas_type"],
                spectral_section["wl_min"],
                spectral_section["wl_max"],
            ],
            outputs=[
                spectral_section["spectral_plot"],
                spectral_section["plot_info"],
            ],
        )

    return interface
