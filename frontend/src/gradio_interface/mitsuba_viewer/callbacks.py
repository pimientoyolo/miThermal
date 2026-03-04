from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from ..components import (
    build_object_group_section,
    build_unified_config_section,
    build_upload_section,
    build_visualization_section,
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
        _ = client.load_scene(scene_path)
        info_text, scene_json, labels = _summarize_scene_and_update_state()
        return None, info_text, scene_json, labels, labels
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
                    y = [1 - (r / 100.0) for r in reflection]
                elif emissivity and wavelengths and len(emissivity) == len(wavelengths):
                    y = emissivity
                else:
                    y = None
                if y:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wavelengths,
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
        </style>
        """
        )
        gr.HTML(
            """
            <h1 style='text-align:center;color:#2e86de;'>🎨 MiThermal</h1>
            """
        )
        with gr.Tabs():
            with gr.Tab("📁 Cargar Escena"):
                upload_section = build_upload_section(show_server_path=False)
            with gr.Tab("🖼️ Visualización"):
                visualization_section = build_visualization_section()
            with gr.Tab("🎯 Objetos y Grupos"):
                og_section = build_object_group_section()
            with gr.Tab("⚙️ Config"):
                config_section = build_unified_config_section()
            with gr.Tab("🎥 Interpolación Cámara"):
                camera_interp_section = build_camera_interpolation_section()
            with gr.Tab("💾 Gestión de Cache"):
                cache_section = build_cache_management_section()
            with gr.Tab("📊 Datos Espectrales"):
                spectral_section = build_spectral_plot_section()

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
                    return (
                        gr.update(), gr.update(), gr.update(),  # spp_k, width, height
                        gr.update(), gr.update(), gr.update(),  # rx, ry, rz
                        gr.update(), gr.update(), gr.update(),  # tx, ty, tz
                        gr.update(),                             # fov
                        gr.update(), gr.update(), gr.update(),   # wl_min, wl_max, bands
                        gr.update(),                              # config_info JSON
                    )
                spp = cfg.get("spp")
                k_val = None
                if isinstance(spp, (int, float)) and spp > 0:
                    # calcular k = log2(spp), entero
                    try:
                        k_val = int(round(math.log2(float(spp))))
                    except Exception:
                        k_val = None
                width = cfg.get("width")
                height = cfg.get("height")
                rx = cfg.get("rotate_x")
                ry = cfg.get("rotate_y")
                rz = cfg.get("rotate_z")
                tx = cfg.get("translate_x")
                ty = cfg.get("translate_y")
                tz = cfg.get("translate_z")
                fov = cfg.get("fov")
                wavelengths = cfg.get("wavelengths") or []
                wl_min = int(min(wavelengths)) if wavelengths else None
                wl_max = int(max(wavelengths)) if wavelengths else None
                bands = cfg.get("num_bands")
                # Construir label dinámico para SPP = 2^k
                spp_label = "SPP (2^k)"
                try:
                    if k_val is not None:
                        spp_label = f"SPP = {int(2 ** int(k_val))}"
                except Exception:
                    pass
                return (
                    gr.update(value=k_val, label=spp_label),
                    gr.update(value=width),
                    gr.update(value=height),
                    gr.update(value=rx),
                    gr.update(value=ry),
                    gr.update(value=rz),
                    gr.update(value=tx),
                    gr.update(value=ty),
                    gr.update(value=tz),
                    gr.update(value=fov),
                    gr.update(value=wl_min),
                    gr.update(value=wl_max),
                    gr.update(value=bands),
                    gr.update(value=cfg),
                )
            except Exception:
                return (
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(),
                )

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
            labels = [label for label, oid in viewer_state.object_id_mapping.items() if oid in oids]
            labels_sorted = sorted(labels)
            if labels_sorted:
                mv, info, _of, _t, plot = view_object_3d(labels_sorted[0])
                return gr.update(choices=labels_sorted, value=[]), gr.update(value=info), mv, plot
            return gr.update(choices=[], value=[]), gr.update(value=f"Sin miembros para {selection}"), None, None

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
            """Actualiza objeto(s) (temperatura + emisividad) usando update_with_emissivity.

            - Objeto: si no se ingresa temperatura, leer la actual del backend.
            - Instancia: igual que objeto para cada miembro (si no hay input, se intenta leer cada una; fallback 300K).
            - Familia: temperatura por defecto 300K salvo que usuario provea otra.
            El archivo de emisividad es obligatorio.
            """
            mode_l = (mode or '').lower()
            client = get_client()
            path = getattr(emissivity_file, 'name', None)
            if not path:
                return gr.update(value='❌ Suba archivo de emisividad')
            # targets
            if mode_l.startswith('obj'):
                oid = viewer_state.object_id_mapping.get(selection)
                targets = [oid] if oid else []
            elif mode_l.startswith('inst'):
                targets = viewer_state.object_groups_instance.get(selection, [])
            else:
                targets = viewer_state.object_groups_family.get(selection, [])
            if not targets:
                return gr.update(value='❌ Nada seleccionado')
            # parse user temp
            try:
                user_temp = float(temp_value) if temp_value not in (None, '') else None
            except Exception:
                user_temp = None
            objs_payload = []
            for oid in targets:
                if mode_l.startswith('fam'):
                    t = user_temp if user_temp is not None else 300.0
                else:
                    if user_temp is not None:
                        t = user_temp
                    else:
                        # obtener del backend
                        try:
                            obj_resp = client.get_object(oid)
                            if obj_resp.get('status') == 'success':
                                odata = obj_resp.get('object', {})
                            else:
                                odata = obj_resp.get('object', obj_resp)
                            if odata and odata.get('temperature') is not None:
                                t = float(odata.get('temperature'))
                            else:
                                t = 300.0
                        except Exception:
                            t = 300.0
                objs_payload.append({'id': oid, 'temperature': t})
            res = client.update_objects_with_emissivity(objs_payload, path)
            if res.get('status') == 'error':
                return gr.update(value=f"❌ Error: {res.get('detail')}")
            msg = res.get('message') if isinstance(res, dict) else 'OK'
            return gr.update(value=f"✅ Actualizados {len(objs_payload)} objeto(s). {msg}")

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
            spp_k,
            width,
            height,
            wl_min,
            wl_max,
            bands,
            air_temperature,
            air_file,
            rx,
            ry,
            rz,
            tx,
            ty,
            tz,
            fov,
        ):
            client = get_client()
            results: Dict[str, Dict] = {}
            # Cámara
            try:
                k = int(float(spp_k)) if spp_k is not None else None
                spp_val = 2 ** k if k is not None else None
                def pos_int(x):
                    return x is None or (isinstance(x, (int, float)) and int(x) > 0)
                if not pos_int(spp_val) or not pos_int(width) or not pos_int(height):
                    results["camera"] = {"status": "error", "detail": "Valores inválidos"}
                else:
                    results["camera"] = client.update_camera_config(
                        spp=int(spp_val) if spp_val is not None else None,
                        width=int(width) if width is not None else None,
                        height=int(height) if height is not None else None,
                        rotate_x=float(rx) if rx is not None else None,
                        rotate_y=float(ry) if ry is not None else None,
                        rotate_z=float(rz) if rz is not None else None,
                        translate_x=float(tx) if tx is not None else None,
                        translate_y=float(ty) if ty is not None else None,
                        translate_z=float(tz) if tz is not None else None,
                        fov=float(fov) if fov is not None else None,
                    )
            except Exception as e:
                results["camera"] = {"status": "error", "detail": str(e)}
            # Espectro
            try:
                wl_min_i = int(wl_min) if wl_min is not None else None
                wl_max_i = int(wl_max) if wl_max is not None else None
                bands_i = int(bands) if bands is not None else None
                if wl_min_i is None or wl_max_i is None or bands_i is None:
                    results["spectrum"] = {"status": "error", "detail": "Campos requeridos"}
                elif wl_min_i < 0 or wl_max_i < 0 or wl_min_i >= wl_max_i or bands_i <= 0:
                    results["spectrum"] = {"status": "error", "detail": "Valores espectro inválidos"}
                else:
                    results["spectrum"] = client.update_wavelengths(wl_min_i, wl_max_i, bands_i)
            except Exception as e:
                results["spectrum"] = {"status": "error", "detail": str(e)}
            # Aire
            air_plot = None
            try:
                # Temperatura del aire (opcional)
                if air_temperature is not None and air_temperature != "":
                    try:
                        t = float(air_temperature)
                        if t > 0:
                            results["air_temperature"] = client.set_air_temperature(t)
                    except Exception as _:
                        results["air_temperature"] = {"status": "error", "detail": "Temperatura inválida"}
                if air_file is not None:
                    path = getattr(air_file, "name", None)
                    if path:
                        results["air"] = client.upload_air_attenuation(path)
                text = client.get_air_attenuation()
                try:
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
            except Exception as e:
                results["air"] = {"status": "error", "detail": str(e)}
            # Estado
            status_lines = []
            for k2, v in results.items():
                try:
                    st = v.get("status", "ok") if isinstance(v, dict) else "ok"
                except Exception:
                    st = "ok"
                status_lines.append(f"{k2}: {st}")
            status_text = "\n".join(status_lines) if status_lines else "Sin cambios"
            return results, status_text, air_plot

        config_section["apply_all_btn"].click(
            fn=apply_all_config_cb,
            inputs=[
                config_section["camera_spp"],
                config_section["camera_width"],
                config_section["camera_height"],
                config_section["wl_min"],
                config_section["wl_max"],
                config_section["bands"],
                config_section["air_temperature"],
                config_section["air_file"],
                config_section["rotate_x"],
                config_section["rotate_y"],
                config_section["rotate_z"],
                config_section["translate_x"],
                config_section["translate_y"],
                config_section["translate_z"],
                config_section["fov"],
            ],
            outputs=[
                config_section["config_info"],
                config_section["config_status"],
                config_section["air_plot"],
            ],
        )


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
            origin_x, origin_y, origin_z,
            end_x, end_y, end_z,
            target_x, target_y, target_z,
            num_steps
        ):
            """Genera interpolación de cámara llamando al endpoint del backend."""
            try:
                client = get_client()
                data = {
                    "origin": [origin_x, origin_y, origin_z],
                    "end": [end_x, end_y, end_z],
                    "tracked_point": [target_x, target_y, target_z],
                    "num_steps": int(num_steps)
                }
                
                # Llamar al endpoint
                response = client.session.post(
                    f"{client.base_url}/scene/camera/interpolation",
                    json=data
                )
                response.raise_for_status()
                frames = response.json()
                
                status_msg = f"✅ Generados {len(frames)} frames de interpolación exitosamente."
                return status_msg, frames
                
            except Exception as e:
                error_msg = f"❌ Error al generar interpolación: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg, []

        def camera_render_animation_cb(
            origin_x, origin_y, origin_z,
            end_x, end_y, end_z,
            target_x, target_y, target_z,
            num_steps
        ):
            """Renderiza la animación completa de cámara."""
            try:
                client = get_client()
                data = {
                    "origin": [origin_x, origin_y, origin_z],
                    "end": [end_x, end_y, end_z],
                    "tracked_point": [target_x, target_y, target_z],
                    "num_steps": int(num_steps)
                }
                
                # Llamar al endpoint de renderizado
                response = client.session.post(
                    f"{client.base_url}/scene/camera/animation/render",
                    json=data,
                    timeout=600  # 10 minutos de timeout
                )
                response.raise_for_status()
                
                # Guardar el ZIP temporalmente
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tf:
                    tf.write(response.content)
                    zip_path = tf.name
                
                final_msg = f"✅ Animación completada: {int(num_steps)} frames renderizados y guardados en ZIP."
                return final_msg, zip_path
                
            except Exception as e:
                error_msg = f"❌ Error al renderizar animación: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg, None

        # Callbacks para gestión de cache
        def refresh_cache_stats_cb():
            """Obtiene y muestra las estadísticas del cache."""
            try:
                client = get_client()
                result = client.get_cache_stats()
                
                if result.get("status") == "success":
                    stats = result.get("data", {})
                    
                    # Formatear mensaje legible
                    hits = stats.get("hits", 0)
                    misses = stats.get("misses", 0)
                    total = hits + misses
                    hit_rate = stats.get("hit_rate", 0.0) * 100
                    size = stats.get("size", 0)
                    max_size = stats.get("max_size", 0)
                    
                    status_msg = (
                        f"✅ Estadísticas actualizadas:\n"
                        f"• Aciertos: {hits} / {total} ({hit_rate:.1f}%)\n"
                        f"• Tamaño: {size} / {max_size} entradas"
                    )
                    
                    return stats, status_msg
                else:
                    error_msg = f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    return {}, error_msg
                    
            except Exception as e:
                error_msg = f"❌ Error al obtener estadísticas: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {}, error_msg

        def clear_cache_cb():
            """Limpia todo el cache de firmas espectrales."""
            try:
                client = get_client()
                result = client.clear_cache()
                
                if result.get("status") == "success":
                    data = result.get("data", {})
                    cleared = data.get("entries_cleared", 0)
                    status_msg = f"✅ Cache limpiado exitosamente. {cleared} entradas eliminadas."
                    
                    # Después de limpiar, obtener estadísticas actualizadas
                    stats_result = client.get_cache_stats()
                    if stats_result.get("status") == "success":
                        return stats_result.get("data", {}), status_msg
                    else:
                        return {}, status_msg
                else:
                    error_msg = f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    return {}, error_msg
                    
            except Exception as e:
                error_msg = f"❌ Error al limpiar cache: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {}, error_msg

        camera_interp_section["generate_btn"].click(
            fn=camera_interpolation_cb,
            inputs=[
                camera_interp_section["origin_x"],
                camera_interp_section["origin_y"],
                camera_interp_section["origin_z"],
                camera_interp_section["end_x"],
                camera_interp_section["end_y"],
                camera_interp_section["end_z"],
                camera_interp_section["target_x"],
                camera_interp_section["target_y"],
                camera_interp_section["target_z"],
                camera_interp_section["num_steps"],
            ],
            outputs=[
                camera_interp_section["status_output"],
                camera_interp_section["interpolation_result"],
            ],
        )

        camera_interp_section["render_btn"].click(
            fn=camera_render_animation_cb,
            inputs=[
                camera_interp_section["origin_x"],
                camera_interp_section["origin_y"],
                camera_interp_section["origin_z"],
                camera_interp_section["end_x"],
                camera_interp_section["end_y"],
                camera_interp_section["end_z"],
                camera_interp_section["target_x"],
                camera_interp_section["target_y"],
                camera_interp_section["target_z"],
                camera_interp_section["num_steps"],
            ],
            outputs=[
                camera_interp_section["status_output"],
                camera_interp_section["animation_zip"],
            ],
        )

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
                
                if plot_type == "Emisividad":
                    if not object_id:
                        return None, "❌ Por favor selecciona un objeto"
                    result = client.get_emissivity_spectrum(object_id)
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    values = data.get("values", [])
                    
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
                    result = client.get_reflectance_spectrum(object_id)
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    values = data.get("values", [])
                    
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
                    result = client.get_atmospheric_spectrum(gas=gas_type)
                    if result.get("status") != "success":
                        return None, f"❌ Error: {result.get('detail', 'Error desconocido')}"
                    
                    data = result.get("data", {})
                    wavelengths = data.get("wavelengths", [])
                    attenuation = data.get("attenuation", [])
                    transmittance = data.get("transmittance", [])
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths, y=attenuation,
                        mode='lines',
                        name='Atenuación',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Atenuación:</b> %{y:.4f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths, y=transmittance,
                        mode='lines',
                        name='Transmitancia',
                        line=dict(color='green', width=2),
                        hovertemplate='<b>Longitud de onda:</b> %{x:.1f} nm<br><b>Transmitancia:</b> %{y:.4f}<extra></extra>'
                    ))
                    
                    plot_info = f"✅ Atmósfera: {gas_type}\nPromedio Atenuación: {np.mean(attenuation):.3f}"
                    title_text = f"Atenuación Atmosférica - {gas_type}"
                    ylabel_text = "Atenuación / Transmitancia"
                
                # Configurar diseño del gráfico
                fig.update_layout(
                    title=title_text,
                    xaxis_title="Longitud de Onda (nm)",
                    yaxis_title=ylabel_text,
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
