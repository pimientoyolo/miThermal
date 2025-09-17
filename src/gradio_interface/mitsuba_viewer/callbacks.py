from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from ..components import (
    build_object_group_section,
    build_unified_config_section,
    build_upload_section,
    build_visualization_section,
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

        base = label_src.split("-", 1)[0]

        m = re.match(r"^(?P<prefix>.+?)_(?P<num>\d+)$", base)
        if m:
            family = m.group("prefix")
            instance = base
        else:
            m2 = re.match(r"^(?P<prefix>.*?)(?P<num>\d+)$", base)
            if m2 and m2.group("prefix"):
                family = m2.group("prefix")
                instance = base
            else:
                family = base
                instance = base

        inst.setdefault(instance, []).append(oid)
        fam.setdefault(family, []).append(oid)
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

        # Emisividad: si hay datos, graficar
        emissivity_plot = None
        try:
            wavelengths = obj.get("wavelengths")
            reflection = obj.get("reflection")
            emissivity = obj.get("emissivity")
            if wavelengths and (reflection or emissivity):
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(4, 3))
                if reflection and not emissivity:
                    y = [1 - (r / 100.0) for r in reflection]
                elif emissivity and wavelengths and len(emissivity) == len(wavelengths):
                    y = emissivity
                else:
                    y = None
                if y:
                    ax.plot(wavelengths, y, label="Emisividad", color="#e67e22")
                    ax.set_xlabel("Longitud de onda")
                    ax.set_ylabel("Emisividad")
                    ax.set_title("Emisividad vs λ")
                    ax.set_ylim(0, 1.05)
                    ax.grid(alpha=0.3)
                    ax.legend()
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
        

        def upload_zip_and_prefill(zip_file):
            # Ejecutar carga de ZIP
            out_img, info_text, scene_json, labels1, labels2 = upload_zip_file(zip_file)
            # Obtener config para prellenar
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
            # Insertar antes del último elemento (config_info)
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

        upload_section["upload_btn"].click(
            fn=upload_zip_and_prefill,
            inputs=[upload_section["zip_file"]],
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
            client = get_client()
            if not file_name:
                return (
                    None,
                    "❌ Seleccione una escena predeterminada",
                    {},
                    [],
                    [],
                    gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    None,
                )
            try:
                _ = client.select_default_scene(file_name)
            except Exception:
                pass
            # Tras seleccionar, refrescar como en load_server_and_prefill (reutilizamos lógica)
            out_img, info_text, scene_json, labels1, labels2 = load_server_scene("")
            cfg = None
            try:
                cfg = get_client().get_camera_config()
            except Exception:
                cfg = None
            updates = list(_prefill_updates_from_config(cfg))
            try:
                air_temp = get_client().get_air_temperature()
            except Exception:
                air_temp = None
            updates.insert(-1, gr.update(value=air_temp))
            try:
                air_suggest_update = air_suggest_cb()
            except Exception:
                air_suggest_update = gr.update(choices=[], value=None)
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

        # Descargar miThermal actual como archivo
        def mithermal_get_cb():
            client = get_client()
            try:
                r = client.get_mithermal_zip()
                if r.get("status") == "ok":
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tf:
                        tf.write(r.get("zip_bytes", b""))
                        return tf.name
            except Exception:
                pass
            return None

        upload_section["mithermal_get_btn"].click(
            fn=mithermal_get_cb,
            inputs=[],
            outputs=[upload_section["zip_file"]],
        )

        # Subir miThermal.zip completo
        def mithermal_post_cb(file):
            client = get_client()
            path = getattr(file, "name", None)
            if not path:
                return gr.update(value="❌ Archivo inválido")
            try:
                res = client.post_mithermal(path)
                msg = res.get("message") if isinstance(res, dict) else "OK"
                return gr.update(value=f"✅ Enviado miThermal: {msg}")
            except Exception as e:
                return gr.update(value=f"❌ Error enviando miThermal: {e}")

        upload_section["mithermal_post_btn"].click(
            fn=mithermal_post_cb,
            inputs=[upload_section["mithermal_upload"]],
            outputs=[upload_section["scene_info"]],
        )
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

        def og_apply_temp(mode: str, selection: str, temp_value):
            if temp_value is None:
                return gr.update(value="❌ Ingrese temperatura")
            mode_l = (mode or "").lower()
            client = get_client()
            targets: List[str] = []
            if mode_l.startswith("obj"):
                oid = viewer_state.object_id_mapping.get(selection)
                if oid:
                    targets = [oid]
            elif mode_l.startswith("inst"):
                targets = viewer_state.object_groups_instance.get(selection, [])
            else:
                targets = viewer_state.object_groups_family.get(selection, [])
            if not targets:
                return gr.update(value="❌ Nada que aplicar")
            ok = 0
            err = 0
            for oid in targets:
                try:
                    r = client.update_object_temperature(oid, float(temp_value))
                    if r.get("status") == "error":
                        err += 1
                    else:
                        ok += 1
                except Exception:
                    err += 1
            msg = f"✅ Temperatura aplicada a {ok}."
            if err:
                msg += f" ⚠️ Fallos: {err}"
            return gr.update(value=msg)

        og_section["apply_temp_btn"].click(
            fn=og_apply_temp,
            inputs=[og_section["mode_radio"], og_section["selector"], og_section["temp_input"]],
            outputs=[og_section["info_text"]],
        )

        def og_apply_emissivity(mode: str, selection: str, emissivity_file):
            if emissivity_file is None:
                return gr.update(value="❌ Suba archivo")
            mode_l = (mode or "").lower()
            client = get_client()
            path = getattr(emissivity_file, "name", None)
            if not path:
                return gr.update(value="❌ Archivo inválido")
            targets: List[str] = []
            if mode_l.startswith("obj"):
                oid = viewer_state.object_id_mapping.get(selection)
                if oid:
                    targets = [oid]
            elif mode_l.startswith("inst"):
                targets = viewer_state.object_groups_instance.get(selection, [])
            else:
                targets = viewer_state.object_groups_family.get(selection, [])
            if not targets:
                return gr.update(value="❌ Nada que aplicar")
            ok = 0
            err = 0
            for oid in targets:
                try:
                    r = client.upload_emissivity_file(oid, path)
                    if r.get("status") == "error":
                        err += 1
                    else:
                        ok += 1
                except Exception:
                    err += 1
            msg = f"✅ Emisividad aplicada a {ok}."
            if err:
                msg += f" ⚠️ Fallos: {err}"
            return gr.update(value=msg)

        og_section["apply_emissivity_btn"].click(
            fn=og_apply_emissivity,
            inputs=[og_section["mode_radio"], og_section["selector"], og_section["emissivity_file"]],
            outputs=[og_section["info_text"]],
        )

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
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(4, 3))
                sc = ax.scatter(wavelengths, emissivity, c=emissivity, cmap="magma")
                ax.set_xlabel("Longitud de onda")
                ax.set_ylabel("Emisividad")
                plt.colorbar(sc, ax=ax, label="Emisividad (preview)")
                ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3)
                ax.set_title("Previsualización Emisividad")
                ax.legend()
                return fig
            except Exception as e:
                logger.warning(f"preview_emissivity_file fallo: {e}")
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
                    import matplotlib.pyplot as plt

                    xs, ys = parse_air_text_to_xy(text)
                    if xs and ys:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(xs, ys, color="#D55E00", linewidth=1.8)
                        ax.set_title("Atenuación del aire")
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.grid(alpha=0.3)
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
                import matplotlib.pyplot as plt
                xs, ys = parse_air_text_to_xy(text)
                if xs and ys:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(xs, ys, color="#D55E00", linewidth=1.8)
                    ax.set_title("Atenuación del aire")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(alpha=0.3)
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
                import matplotlib.pyplot as plt
                xs, ys = parse_air_text_to_xy(text)
                if xs and ys:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(xs, ys, color="#D55E00", linewidth=1.8)
                    ax.set_title("Atenuación del aire")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(alpha=0.3)
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

    return interface
