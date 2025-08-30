"""API client para comunicarse con el backend FastAPI de Mitsuba."""

from __future__ import annotations
from pathlib import Path
import logging
import base64
import requests


logger = logging.getLogger(__name__)



class MitsubaAPIClient:
    """Cliente para conectar con la API FastAPI del servidor Mitsuba.

    Endpoints soportados (v2):
        POST /scene/load_scene -> sube zip y devuelve imagen (png)
        GET  /scene/render_scene_rgb -> fuerza render RGB de la escena cargada
        GET  /scene/has_loaded_scene -> bool
        GET  /object/suggest_objects -> lista [{id,suggest}]
        GET  /object/object?id=... -> stream de archivo (.obj/.ply)

    Algunos endpoints legacy (/objects, /download, /scene-info) quedaron obsoletos.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    def upload_scene(self, zip_file_path: str):
        """Sube archivo ZIP de escena (POST /scene/load_scene) y retorna dict con imagen base64."""
        with open(zip_file_path, "rb") as f:
            files = {"file": (Path(zip_file_path).name, f, "application/zip")}
            resp = self.session.post(f"{self.base_url}/scene/load_scene", files=files)
            resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        if "image" in ctype:
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
            return {"status": "ok", "image_base64": img_b64}
        # fallback: devolver bytes como base64 aun si no marcó image
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        return {"status": "ok", "image_base64": img_b64}

    def render_scene_rgb(self):
        """Fuerza render (GET /scene/render_scene_rgb) y retorna imagen base64."""
        resp = self.session.get(f"{self.base_url}/scene/render_scene_rgb")
        resp.raise_for_status()
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        return {"status": "ok", "image_base64": img_b64}

    def has_loaded_scene(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/scene/has_loaded_scene")
            r.raise_for_status()
            return r.json() if r.headers.get("content-type","application/json").startswith("application/json") else bool(r.text.strip() == 'true')
        except Exception as e:
            logger.error(f"has_loaded_scene error: {e}")
            return False

    # --- NUEVOS ENDPOINTS OBJETOS ---
    def suggest_objects(self):
        try:
            r = self.session.get(f"{self.base_url}/object/suggest_objects")
            r.raise_for_status()
            return r.json()  # lista de {id,suggest}
        except Exception as e:
            logger.error(f"suggest_objects error: {e}")
            return []

    def download_object(self, object_id: str, save_path: str) -> str:
        """Descarga objeto (.obj o .ply convertido) desde /object/object?id=..."""
        try:
            r = self.session.get(f"{self.base_url}/object/object", params={"id": object_id})
            r.raise_for_status()
            # Intenta deducir extensión a partir de header filename
            dispo = r.headers.get("content-disposition", "")
            ext = ".obj"
            if "filename=" in dispo:
                fn = dispo.split("filename=")[-1].strip('"')
                pext = Path(fn).suffix.lower()
                if pext:
                    ext = pext
            save_path = Path(save_path)
            # Asegurar que el directorio exista
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix.lower() != ext:
                save_path = save_path.with_suffix(ext)
            save_path.write_bytes(r.content)
            return str(save_path)
        except Exception as e:
            logger.error(f"download_object error: {e}")
            return ""

