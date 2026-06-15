"""API client para comunicarse con el backend FastAPI de Mitsuba."""

from __future__ import annotations
from pathlib import Path
import logging
import base64
import requests
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = os.environ.get("MITSUBA_API_BASE", os.environ.get("MITHERMAL_BACKEND", "http://localhost:8000"))


def _get_downloads_dir() -> Path:
    import tempfile
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "frontend").is_dir() and (parent / "backend").is_dir():
            d = parent / "downloads"
            d.mkdir(parents=True, exist_ok=True)
            return d
    return Path(tempfile.gettempdir())

DOWNLOADS_DIR = _get_downloads_dir()



def set_default_base_url(url: str) -> None:
    """Establece la URL base por defecto usada al crear nuevos clientes.

    Esto permite que la aplicación (p.ej. `frontend/main.py`) fije el backend
    mediante argumentos de línea de comandos antes de crear componentes.
    """
    global DEFAULT_BASE_URL
    DEFAULT_BASE_URL = url.rstrip("/")


class MitsubaAPIClient:
    """Cliente para conectar con la API FastAPI del servidor Mitsuba."""

    def __init__(self, base_url: str | None = None):
        # Prioridad: argumento > variable de entorno > valor por defecto
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = (os.environ.get("MITSUBA_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
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

            # Guardar con extensión real recibida (obj/ply)
            ext = Path(object_id).suffix.lower() or ".obj"
            disposition = r.headers.get("Content-Disposition", "")
            if "filename=" in disposition:
                file_name = disposition.split("filename=", 1)[1].strip().strip('"')
                detected_ext = Path(file_name).suffix.lower()
                if detected_ext:
                    ext = detected_ext

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path = save_path.with_suffix(ext)
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

    def download_config(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/config/download")
            r.raise_for_status()
            return {"status": "ok", "json_bytes": r.content}
        except Exception as e:
            logger.error(f"Download config error: {e}")
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
        theta: float | None = None,
        phi: float | None = None,
        radius: float | None = None,
        target_x: float | None = None,
        target_y: float | None = None,
        target_z: float | None = None,
    ) -> Dict:
        try:
            payload = {}
            if spp is not None: payload["spp"] = int(spp)
            if width is not None: payload["width"] = int(width)
            if height is not None: payload["height"] = int(height)
            if rotate_x is not None: payload["rotate_x"] = float(rotate_x)
            if rotate_y is not None: payload["rotate_y"] = float(rotate_y)
            if rotate_z is not None: payload["rotate_z"] = float(rotate_z)
            if translate_x is not None: payload["translate_x"] = float(translate_x)
            if translate_y is not None: payload["translate_y"] = float(translate_y)
            if translate_z is not None: payload["translate_z"] = float(translate_z)
            if fov is not None: payload["fov"] = float(fov)
            if theta is not None: payload["theta"] = float(theta)
            if phi is not None: payload["phi"] = float(phi)
            if radius is not None: payload["radius"] = float(radius)
            if target_x is not None: payload["target_x"] = float(target_x)
            if target_y is not None: payload["target_y"] = float(target_y)
            if target_z is not None: payload["target_z"] = float(target_z)
            
            r = self.session.put(f"{self.base_url}/config/camera", json=payload)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
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

    def update_wavelengths(self, wavelength_min: float, wavelength_max: float, bands: int) -> Dict:
        """Actualiza las longitudes de onda y número de bandas.

        Endpoint: PUT /config/wavelengths (query params)
        """
        try:
            params = {
                "wavelength_min": float(wavelength_min),
                "wavelength_max": float(wavelength_max),
                "bands": int(bands),
            }
            r = self.session.put(f"{self.base_url}/config/wavelengths", params=params)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
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
                ctype = r.headers.get("Content-Type", "").lower()
                if "image" in ctype:
                    import base64
                    img_b64 = base64.b64encode(r.content).decode("utf-8")
                    return {"status": "ok", "image_base64": img_b64}
                # Intentar JSON
                try:
                    data = r.json()
                    return {"status": "success", "message": data}
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

    # ---------------------- Cache Management ----------------------
    def get_cache_stats(self) -> Dict:
        """GET /config/cache/stats
        
        Obtiene estadísticas del cache de emisión/reflectancia.
        Retorna: dict con hits, misses, hit_rate, size, max_size, etc.
        """
        try:
            r = self.session.get(f"{self.base_url}/config/cache/stats")
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Get cache stats error: {e}")
            return {"status": "error", "detail": str(e)}

    def clear_cache(self) -> Dict:
        """POST /config/cache/clear
        
        Limpia todo el cache de emisión/reflectancia.
        Retorna: dict con mensaje de confirmación.
        """
        try:
            r = self.session.post(f"{self.base_url}/config/cache/clear")
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Clear cache error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_emissivity_map_config(self) -> Dict:
        """GET /config/emissivity-map"""
        try:
            r = self.session.get(f"{self.base_url}/config/emissivity-map")
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Get emissivity map config error: {e}")
            return {"status": "error", "detail": str(e)}

    def update_emissivity_map_config(self, use_custom: bool, wl_min: float, wl_max: float, bands: int) -> Dict:
        """PUT /config/emissivity-map"""
        try:
            payload = {
                "use_custom": use_custom,
                "wl_min": wl_min,
                "wl_max": wl_max,
                "bands": bands
            }
            r = self.session.put(f"{self.base_url}/config/emissivity-map", json=payload)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Update emissivity map config error: {e}")
            return {"status": "error", "detail": str(e)}

    def download_full_config(self) -> bytes:
        """GET /config/full/download -> Returns zip bytes"""
        r = self.session.get(f"{self.base_url}/config/full/download")
        r.raise_for_status()
        return r.content

    def upload_full_config(self, zip_path: str) -> Dict:
        """POST /config/full/upload"""
        try:
            with open(zip_path, 'rb') as f:
                files = {"file": (os.path.basename(zip_path), f, "application/zip")}
                r = self.session.post(f"{self.base_url}/config/full/upload", files=files)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Upload full config error: {e}")
            return {"status": "error", "detail": str(e)}

    # ---------------------- Spectral Data Export ----------------------
    def get_object_spectral_data(
        self, 
        object_id: str,
        wavelength_min_nm: float = None,
        wavelength_max_nm: float = None
    ) -> Dict:
        try:
            params = {}
            if wavelength_min_nm is not None: params["wavelength_min_nm"] = wavelength_min_nm
            if wavelength_max_nm is not None: params["wavelength_max_nm"] = wavelength_max_nm
            
            # Codificar object_id para la URL (manejar slashes como parte de la ruta)
            import urllib.parse
            safe_id = urllib.parse.quote(object_id, safe='')
            
            r = self.session.get(f"{self.base_url}/spectral/object/{safe_id}", params=params)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Get object spectral data error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_atmospheric_spectrum(
        self, 
        gas: str = "air",
        wavelength_min_nm: int = None,
        wavelength_max_nm: int = None
    ) -> Dict:
        try:
            params = {"gas": gas}
            if wavelength_min_nm is not None: params["wavelength_min_nm"] = wavelength_min_nm
            if wavelength_max_nm is not None: params["wavelength_max_nm"] = wavelength_max_nm
            r = self.session.get(f"{self.base_url}/spectral/atmosphere", params=params)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Get atmospheric spectrum error: {e}")
            return {"status": "error", "detail": str(e)}

    def get_blackbody_spectrum(
        self, 
        temperature_k: float = 300.0,
        wavelength_min_nm: float = 8000.0,
        wavelength_max_nm: float = 12000.0,
        num_points: int = 100
    ) -> Dict:
        try:
            params = {
                "temperature_k": temperature_k,
                "wavelength_min_nm": wavelength_min_nm,
                "wavelength_max_nm": wavelength_max_nm,
                "num_points": num_points
            }
            r = self.session.get(f"{self.base_url}/spectral/blackbody", params=params)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Get blackbody spectrum error: {e}")
            return {"status": "error", "detail": str(e)}

    def interpolate_spectral_data(
        self,
        wavelengths_source: list[float],
        values_source: list[float],
        wavelengths_target: list[float]
    ) -> Dict:
        try:
            payload = {
                "wavelengths_source": wavelengths_source,
                "values_source": values_source,
                "wavelengths_target": wavelengths_target
            }
            r = self.session.post(f"{self.base_url}/spectral/interpolate", json=payload)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Interpolate spectral data error: {e}")
            return {"status": "error", "detail": str(e)}

    def render_thermal(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/render/thermal")
            r.raise_for_status()
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

    def render_emissivity(self) -> Dict:
        try:
            r = self.session.get(f"{self.base_url}/render/emissivity/map")
            r.raise_for_status()
            return {"status": "ok", "npy_bytes": r.content}
        except Exception as e:
            logger.error(f"Render emissivity error: {e}")
            return {"status": "error", "detail": str(e)}

    # ---------------------- Actualización de Objetos con Modo ----------------------
    def update_object_with_mode(
        self, 
        object_id: str, 
        mode: str, 
        temperature: float = None, 
        emissivity_file_path: str = None,
        is_reflectance: bool = False,
        temp_min: float = None,
        temp_max: float = None,
        material_type: str = None,
        roughness: float = None
    ) -> Dict:
        """PUT /object/update-with-mode"""
        try:
            params = {"object_id": object_id, "mode": mode, "is_reflectance": is_reflectance}
            if temperature is not None: params["temperature"] = temperature
            if temp_min is not None: params["temp_min"] = temp_min
            if temp_max is not None: params["temp_max"] = temp_max
            if material_type is not None: params["material_type"] = material_type
            if roughness is not None: params["roughness"] = roughness
                
            files = None
            f = None
            if emissivity_file_path:
                f = open(emissivity_file_path, 'rb')
                files = {"emissivity_file": (Path(emissivity_file_path).name, f, "text/plain")}
            
            r = self.session.put(f"{self.base_url}/object/update-with-mode", params=params, files=files)
            if f: f.close()
                
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Update object with mode error: {e}")
            return {"status": "error", "detail": str(e)}

    # ---------------------- Configuración Espacial de Cámara ----------------------
    def export_camera_spatial_config(self) -> Dict:
        """GET /config/camera/spatial/export"""
        try:
            r = self.session.get(f"{self.base_url}/config/camera/spatial/export")
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Export camera spatial config error: {e}")
            return {"status": "error", "detail": str(e)}

    def import_camera_spatial_config(self, spatial_config: Dict) -> Dict:
        """POST /config/camera/spatial/import"""
        try:
            r = self.session.post(f"{self.base_url}/config/camera/spatial/import", json=spatial_config)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Import camera spatial config error: {e}")
            return {"status": "error", "detail": str(e)}

    # ---------------------- Animaciones de Cámara ----------------------
    def generate_spherical_interpolation(
        self,
        start_theta: float,
        end_theta: float,
        start_azimuth: float,
        end_azimuth: float,
        tracked_point: list[float],
        radius: float = None,
        start_radius: float = None,
        end_radius: float = None,
        num_steps: int = 30,
        lock_azimuth_to_end: bool = False,
        theta_expr: str | None = None,
        azimuth_expr: str | None = None,
        radius_expr: str | None = None,
        auto_fov: bool = False,
        initial_fov: float | None = None
    ) -> Dict:
        """POST /scene/camera/interpolation/spherical"""
        try:
            payload = {
                "start_theta": start_theta,
                "end_theta": end_theta,
                "start_azimuth": start_azimuth,
                "end_azimuth": end_azimuth,
                "tracked_point": tracked_point,
                "num_steps": num_steps,
                "lock_azimuth_to_end": lock_azimuth_to_end,
                "auto_fov": auto_fov
            }
            if radius is not None: payload["radius"] = radius
            if start_radius is not None: payload["start_radius"] = start_radius
            if end_radius is not None: payload["end_radius"] = end_radius
            if theta_expr is not None: payload["theta_expr"] = theta_expr
            if azimuth_expr is not None: payload["azimuth_expr"] = azimuth_expr
            if radius_expr is not None: payload["radius_expr"] = radius_expr
            if initial_fov is not None: payload["initial_fov"] = initial_fov
            
            r = self.session.post(f"{self.base_url}/scene/camera/interpolation/spherical", json=payload)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Generate spherical interpolation error: {e}")
            return {"status": "error", "detail": str(e)}

    def render_spherical_animation(
        self,
        start_theta: float,
        end_theta: float,
        start_azimuth: float,
        end_azimuth: float,
        tracked_point: list[float],
        radius: float = None,
        start_radius: float = None,
        end_radius: float = None,
        num_steps: int = 30,
        lock_azimuth_to_end: bool = False,
        theta_expr: str | None = None,
        azimuth_expr: str | None = None,
        radius_expr: str | None = None,
        auto_fov: bool = False,
        initial_fov: float | None = None,
        spp: int | None = None,
        width: int | None = None,
        height: int | None = None,
        num_bands: int | None = None
    ) -> Dict:
        """POST /scene/camera/animation/render/spherical"""
        try:
            payload = {
                "start_theta": start_theta,
                "end_theta": end_theta,
                "start_azimuth": start_azimuth,
                "end_azimuth": end_azimuth,
                "tracked_point": tracked_point,
                "num_steps": num_steps,
                "lock_azimuth_to_end": lock_azimuth_to_end,
                "auto_fov": auto_fov
            }
            if radius is not None: payload["radius"] = radius
            if start_radius is not None: payload["start_radius"] = start_radius
            if end_radius is not None: payload["end_radius"] = end_radius
            if theta_expr is not None: payload["theta_expr"] = theta_expr
            if azimuth_expr is not None: payload["azimuth_expr"] = azimuth_expr
            if radius_expr is not None: payload["radius_expr"] = radius_expr
            if initial_fov is not None: payload["initial_fov"] = initial_fov
            
            if spp is not None: payload["spp"] = int(spp)
            if width is not None: payload["width"] = int(width)
            if height is not None: payload["height"] = int(height)
            if num_bands is not None: payload["num_bands"] = int(num_bands)
            
            r = self.session.post(f"{self.base_url}/scene/camera/animation/render/spherical", json=payload)
            r.raise_for_status()
            return {"status": "ok", "zip_bytes": r.content}
        except Exception as e:
            logger.error(f"Render spherical animation error: {e}")
            return {"status": "error", "detail": str(e)}

    def preview_spherical_animation(
        self,
        start_theta: float,
        end_theta: float,
        start_azimuth: float,
        end_azimuth: float,
        tracked_point: list[float],
        radius: float = None,
        start_radius: float = None,
        end_radius: float = None,
        num_steps: int = 30,
        lock_azimuth_to_end: bool = False,
        theta_expr: str | None = None,
        azimuth_expr: str | None = None,
        radius_expr: str | None = None,
        auto_fov: bool = False,
        initial_fov: float | None = None
    ) -> Dict:
        """POST /scene/camera/animation/preview/spherical"""
        try:
            payload = {
                "start_theta": start_theta,
                "end_theta": end_theta,
                "start_azimuth": start_azimuth,
                "end_azimuth": end_azimuth,
                "tracked_point": tracked_point,
                "num_steps": num_steps,
                "lock_azimuth_to_end": lock_azimuth_to_end,
                "auto_fov": auto_fov
            }
            if radius is not None: payload["radius"] = radius
            if start_radius is not None: payload["start_radius"] = start_radius
            if end_radius is not None: payload["end_radius"] = end_radius
            if theta_expr is not None: payload["theta_expr"] = theta_expr
            if azimuth_expr is not None: payload["azimuth_expr"] = azimuth_expr
            if radius_expr is not None: payload["radius_expr"] = radius_expr
            if initial_fov is not None: payload["initial_fov"] = initial_fov
            
            r = self.session.post(f"{self.base_url}/scene/camera/animation/preview/spherical", json=payload)
            r.raise_for_status()
            return {"status": "ok", "gif_bytes": r.content}
        except Exception as e:
            logger.error(f"Preview spherical animation error: {e}")
            return {"status": "error", "detail": str(e)}

    def export_camera_animation(self, mode: str, data: Dict) -> Dict:
        """POST /scene/camera/animation/export"""
        try:
            payload = {"mode": mode}
            if mode == "linear": payload["linear_data"] = data
            else: payload["spherical_data"] = data
                
            r = self.session.post(f"{self.base_url}/scene/camera/animation/export", json=payload)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Export camera animation error: {e}")
            return {"status": "error", "detail": str(e)}

    def load_camera_animation(self, animation_config: Dict) -> Dict:
        """POST /scene/camera/animation/load"""
        try:
            r = self.session.post(f"{self.base_url}/scene/camera/animation/load", json=animation_config)
            r.raise_for_status()
            return {"status": "success", "data": r.json()}
        except Exception as e:
            logger.error(f"Load camera animation error: {e}")
            return {"status": "error", "detail": str(e)}

    def download_simulation_zip(self) -> Dict:
        """GET /render/simulation/zip -> Descarga el ZIP de resultados directamente"""
        try:
            r = self.session.get(f"{self.base_url}/render/simulation/zip", stream=True)
            r.raise_for_status()
            import tempfile
            import os
            fd, path = tempfile.mkstemp(suffix=".zip", dir=str(DOWNLOADS_DIR))
            with os.fdopen(fd, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return {"status": "ok", "zip_path": path}
        except Exception as e:
            logger.error(f"Download simulation zip error: {e}")
            return {"status": "error", "detail": str(e)}
