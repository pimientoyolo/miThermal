"""API client para comunicarse con el backend FastAPI de Mitsuba."""

from __future__ import annotations
from pathlib import Path
import logging
import base64
import requests
from typing import Dict

logger = logging.getLogger(__name__)


class MitsubaAPIClient:
    def render_thermal(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/render/thermal")
            r.raise_for_status()
            # Espera un archivo .npy, lo decodifica como base64 para Gradio
            return {"status": "ok", "npy_bytes": r.content}
        except Exception as e:
            logger.error(f"Render thermal error: {e}")
            return {"status": "error", "detail": str(e)}

    def render_depth(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/render/depth")
            r.raise_for_status()
            return {"status": "ok", "npy_bytes": r.content}
        except Exception as e:
            logger.error(f"Render depth error: {e}")
            return {"status": "error", "detail": str(e)}
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
            resp = requests.post(f"{self.base_url}/scene/load", files=files)
            resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        if "image" in ctype:
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
            return {"status": "ok", "image_base64": img_b64}
        return resp.json()

    def load_scene(self, scene_path: str) -> Dict:
        try:
            r = self.session.post(
                f"{self.base_url}/scene/load", params={"scene_path": scene_path}
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Load scene error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_objects(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/object/suggest")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Get objects error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_object(self, object_id: str) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/object/id?object_id={object_id}")
            r.raise_for_status()
            data = r.json()
            # Algunos endpoints devuelven directamente el objeto sin envoltorio {'status': 'success', 'object': {...}}
            if isinstance(data, dict) and 'status' not in data and 'id' in data:
                return {"status": "success", "object": data}
            return data
        except Exception as e:
            logger.error(f"Get object {object_id} error: {e}")
            return {"status": "error", "detail": str(e)}

    def download_object(self, object_id: str, save_path: str) -> str:
        try:
            r = self.session.get(f"{self.base_url}/object/file/id?object_id={object_id}")
            r.raise_for_status()
            # Guardar SIEMPRE como .obj, el servidor ya convierte de .ply a .obj
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path = save_path.with_suffix(".obj")
            save_path.write_bytes(r.content)
            return str(save_path)
        except Exception as e:
            logger.error(f"Download object {object_id} error: {e}")
            return ""

    def update_object_temperature(self, object_id: str, temperature: float) -> Dict:
        """Actualiza la temperatura de un objeto.

        Enpoint: PUT /object/id
        Body esperado: {"id": object_id, "temperature": <float/int>}
        """
        try:
            payload = {"id": object_id, "temperature": temperature}
            r = self.session.put(f"{self.base_url}/object/id", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Update object temperature error (id={object_id}): {e}")
            return {"status": "error", "detail": str(e)}

    def get_emissivity_file(self, object_id: str) -> str:
        """Obtiene el archivo de emisividad (texto) para un objeto.

        Endpoint: GET /object/emissivity/id?object_id=<id>
        Devuelve texto (tbs) con columnas: wavelength reflectance
        """
        try:
            r = self.session.get(f"{self.base_url}/object/emissivity/id", params={"object_id": object_id})
            r.raise_for_status()
            # Puede venir como texto plano
            return r.text
        except Exception as e:
            logger.error(f"Get emissivity file error (id={object_id}): {e}")
            return ""

    def update_object_info(self, object_id: str, **fields) -> Dict:
        """Actualiza campos del objeto (temperatura, curvas, etc.).

        Acepta campos arbitrarios soportados por el backend: temperature, emissivity, reflection, wavelengths
        """
        try:
            payload = {"id": object_id}
            payload.update(fields)
            r = self.session.put(f"{self.base_url}/object/id", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Update object info error (id={object_id}): {e}")
            return {"status": "error", "detail": str(e)}

    def upload_emissivity_file(self, object_id: str, file_path: str) -> Dict:
        """Sube archivo de emisividad usando endpoint multipart.

        Endpoint: PUT /object/emissivity/id?object_id=<id>
        form-data: file=<archivo>
        """
        try:
            with open(file_path, 'rb') as f:
                files = {"file": (Path(file_path).name, f, "text/plain")}
                r = self.session.put(
                    f"{self.base_url}/object/emissivity/id",
                    params={"object_id": object_id},
                    files=files,
                )
                r.raise_for_status()
                # Puede devolver texto o JSON simple
                try:
                    return {"status": "success", "message": r.json()}
                except Exception:
                    return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Upload emissivity file error (id={object_id}): {e}")
            return {"status": "error", "detail": str(e)}

    def get_scene_info(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/scene-info")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Scene info error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_camera_config(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/config/camera")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Get camera config error: {e}")
            return {"status": "error", "detail": str(e)}

    def update_camera_config(
        self,
        spp: int | None = None,
        width: int | None = None,
        height: int | None = None,
        rotate_x: float | None = None,
        rotate_y: float | None = None,
        rotate_z: float | None = None,
        translate_x: float | None = None,
        translate_y: float | None = None,
        translate_z: float | None = None,
        fov: float | None = None,
    ) -> Dict:
        try:
            payload = {}
            if spp is not None:
                payload["spp"] = int(spp)
            if width is not None:
                payload["width"] = int(width)
            if height is not None:
                payload["height"] = int(height)
            if rotate_x is not None:
                payload["rotate_x"] = float(rotate_x)
            if rotate_y is not None:
                payload["rotate_y"] = float(rotate_y)
            if rotate_z is not None:
                payload["rotate_z"] = float(rotate_z)
            if translate_x is not None:
                payload["translate_x"] = float(translate_x)
            if translate_y is not None:
                payload["translate_y"] = float(translate_y)
            if translate_z is not None:
                payload["translate_z"] = float(translate_z)
            if fov is not None:
                payload["fov"] = float(fov)
            r = self.session.put(f"{self.base_url}/config/camera", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Update camera config error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_air_attenuation(self) -> str:
        """Obtiene la atenuación del aire como texto.

        Endpoint: GET /config/air/attenuation
        """
        try:
            r = self.session.get(f"{self.base_url}/config/air/attenuation")
            r.raise_for_status()
            return r.text
        except Exception as e:
            logger.error(f"Get air attenuation error: {e}")
            return ""

    def upload_air_attenuation(self, file_path: str) -> dict:
        """Sube archivo de atenuación del aire (multipart). Endpoint PUT /config/air/attenuation"""
        try:
            with open(file_path, 'rb') as f:
                files = {"file": (Path(file_path).name, f, "text/plain")}
                r = self.session.put(f"{self.base_url}/config/air/attenuation", files=files)
                r.raise_for_status()
                try:
                    return {"status": "success", "message": r.json()}
                except Exception:
                    return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Upload air attenuation error: {e}")
            return {"status": "error", "detail": str(e)}

    def suggest_air_attenuation(self) -> list[str] | Dict:
        try:
            r = self.session.get(f"{self.base_url}/config/air/suggest/attenuation")
            r.raise_for_status()
            data = r.json()
            return data
        except Exception as e:
            logger.error(f"Suggest air attenuation error: {e}")
            return {"status": "error", "detail": str(e)}

    def set_air_attenuation_by_filename(self, file_name: str) -> Dict:
        try:
            r = self.session.put(f"{self.base_url}/config/air/attenuation/filename", params={"file_name": file_name})
            r.raise_for_status()
            try:
                return {"status": "success", "message": r.json()}
            except Exception:
                return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Set air attenuation by filename error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_air_temperature(self) -> float | None:
        """Obtiene la temperatura del aire.

        Endpoint: GET /config/air/temperature
        Puede devolver JSON o texto; se intenta parsear a float.
        """
        try:
            r = self.session.get(f"{self.base_url}/config/air/temperature")
            r.raise_for_status()
            # Intentar JSON primero
            try:
                data = r.json()
                # data puede ser un número directo o un dict
                if isinstance(data, (int, float)):
                    return float(data)
                if isinstance(data, dict):
                    # buscar bajo una clave común
                    for k in ("temperature", "air_temperature", "value"):
                        if k in data and isinstance(data[k], (int, float)):
                            return float(data[k])
            except Exception:
                pass
            # Fallback a texto
            txt = (r.text or "").strip()
            try:
                return float(txt)
            except Exception:
                return None
        except Exception as e:
            logger.error(f"Get air temperature error: {e}")
            return None

    def update_wavelengths(self, wavelength_min: int, wavelength_max: int, bands: int) -> Dict:
        """Actualiza las longitudes de onda y número de bandas.

        Endpoint: PUT /config/wavelengths (query params)
        """
        try:
            params = {
                "wavelength_min": int(wavelength_min),
                "wavelength_max": int(wavelength_max),
                "bands": int(bands),
            }
            r = self.session.put(f"{self.base_url}/config/wavelengths", params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Update wavelengths error: {e}")
            return {"status": "error", "detail": str(e)}

    def set_air_temperature(self, temperature: float) -> Dict:
        """Configura la temperatura del aire en Kelvin.

        Endpoint: PUT /config/air/temperature (query param: temperature)
        """
        try:
            r = self.session.put(
                f"{self.base_url}/config/air/temperature",
                params={"temperature": float(temperature)},
            )
            r.raise_for_status()
            try:
                return {"status": "success", "message": r.json()}
            except Exception:
                return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Set air temperature error: {e}")
            return {"status": "error", "detail": str(e)}

    # ---------------------- Escenas miThermal y default ----------------------
    def get_mithermal_zip(self) -> Dict:
        """Descarga el ZIP completo de la escena actual/seleccionada.

        Endpoint: GET /scene/miThermal
        Devuelve bytes del ZIP o error.
        """
        try:
            r = self.session.get(f"{self.base_url}/scene/miThermal")
            r.raise_for_status()
            return {"status": "ok", "zip_bytes": r.content}
        except Exception as e:
            logger.error(f"Get miThermal.zip error: {e}")
            return {"status": "error", "detail": str(e)}

    def post_mithermal(self, file_path: str) -> Dict:
        """Sube una escena completa (ZIP) para miThermal.

        Endpoint: POST /scene/miThermal (multipart file)
        """
        try:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f, "application/zip")}
                r = self.session.post(f"{self.base_url}/scene/miThermal", files=files)
                r.raise_for_status()
                try:
                    return {"status": "success", "message": r.json()}
                except Exception:
                    return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Post miThermal error: {e}")
            return {"status": "error", "detail": str(e)}

    def suggest_default_scenes(self) -> list[str] | Dict:
        """Obtiene la lista de escenas predeterminadas.

        Endpoint: GET /scene/suggest
        """
        try:
            r = self.session.get(f"{self.base_url}/scene/suggest")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Suggest default scenes error: {e}")
            return {"status": "error", "detail": str(e)}

    def select_default_scene(self, file_name: str) -> Dict:
        """Selecciona una escena predeterminada como actual.

        Endpoint: POST /scene/select/default?file_name=<name>
        """
        try:
            r = self.session.post(f"{self.base_url}/scene/select/default", params={"file_name": file_name})
            r.raise_for_status()
            try:
                return {"status": "success", "message": r.json()}
            except Exception:
                return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Select default scene error (file_name={file_name}): {e}")
            return {"status": "error", "detail": str(e)}

    # ---------------------- Batch update objetos + emisividad ----------------------
    def update_objects_with_emissivity(self, objects: list[dict], emissivity_file_path: str) -> Dict:
        """PUT /object/update-with-emissivity

        Envía una lista de objetos (id, temperature) y un archivo de emisividad.
        Form fields:
          - object_data_json: JSON serializado de la lista
          - emissivity_file: archivo de emisividad (.tbs, .txt)
        """
        import json
        try:
            payload_json = json.dumps(objects)
            with open(emissivity_file_path, 'rb') as f:
                files = {
                    'object_data_json': (None, payload_json, 'application/json'),
                    'emissivity_file': (Path(emissivity_file_path).name, f, 'text/plain'),
                }
                r = self.session.put(f"{self.base_url}/object/update-with-emissivity", files=files)
                r.raise_for_status()
                try:
                    return {"status": "success", "message": r.json()}
                except Exception:
                    return {"status": "success", "message": r.text}
        except Exception as e:
            logger.error(f"Batch update objects with emissivity error: {e}")
            return {"status": "error", "detail": str(e)}
