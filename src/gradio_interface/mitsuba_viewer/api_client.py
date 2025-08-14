"""API client para comunicarse con el backend FastAPI de Mitsuba."""

from __future__ import annotations
from pathlib import Path
import logging
import base64
import requests
from typing import Dict

logger = logging.getLogger(__name__)


class MitsubaAPIClient:
    """Cliente para conectar con la API FastAPI del servidor Mitsuba."""

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
        """Sube ZIP y devuelve imagen (base64) o JSON."""
        with open(zip_file_path, "rb") as f:
            files = {"file": (zip_file_path, f, "application/zip")}
            resp = requests.post(f"{self.base_url}/scene/load_scene", files=files)
            resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        if "image" in ctype:
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
            return {"status": "ok", "image_base64": img_b64}
        return resp.json()

    def load_scene(self, scene_path: str) -> Dict:
        try:
            r = self.session.post(
                f"{self.base_url}/load-scene", params={"scene_path": scene_path}
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Load scene error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_objects(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/objects")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Get objects error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_object(self, object_id: str) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/objects/{object_id}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Get object {object_id} error: {e}")
            return {"status": "error", "detail": str(e)}

    def download_object(self, object_id: str, save_path: str) -> str:
        try:
            r = self.session.get(f"{self.base_url}/download/{object_id}")
            r.raise_for_status()
            obj_info = self.get_object(object_id)
            original_ext = ".obj"
            if obj_info.get("status") == "success":
                ofn = obj_info["object"].get("filename", "")
                if ofn:
                    original_ext = Path(ofn).suffix.lower()
            save_path = Path(save_path)
            if save_path.suffix.lower() != original_ext:
                save_path = save_path.with_suffix(original_ext)
            save_path.write_bytes(r.content)
            return str(save_path)
        except Exception as e:
            logger.error(f"Download object {object_id} error: {e}")
            return ""

    def get_scene_info(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/scene-info")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Scene info error: {e}")
            return {"status": "error", "detail": str(e)}
