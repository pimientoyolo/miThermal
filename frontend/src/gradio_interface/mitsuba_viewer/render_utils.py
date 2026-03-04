from __future__ import annotations
import tempfile
from typing import Callable, Tuple, Optional
from io import BytesIO

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .api_client import MitsubaAPIClient

# --------------------------------------------------------------------------------------
# Utilidades de renderización reutilizables
# --------------------------------------------------------------------------------------

def get_client_instance(get_client_func: Callable[[], MitsubaAPIClient]) -> MitsubaAPIClient:
    """Wrapper para obtener instancia (inyectable en tests)."""
    return get_client_func()


def normalize_to_uint8(arr: np.ndarray) -> Image.Image:
    """Convierte un array 2D en una imagen coloreada con 'magma' e incluye colorbar.

    - Normaliza ignorando NaNs con nanmin/nanmax.
    - Renderiza con matplotlib (imshow + colorbar) y exporta a PNG.
    """
    if arr.ndim != 2:
        raise ValueError("Se esperaba array 2D para normalización a imagen")
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    im = ax.imshow(arr, cmap='magma', vmin=float(min_v), vmax=float(max_v))
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.1)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    pil = Image.open(buf).copy()
    buf.close()
    return pil


def pick_first_band_if_needed(arr: np.ndarray) -> np.ndarray:
    """Devuelve el array 2D apropiado para visualización.

    - Si es 3D (H, W, C) se toma la primera banda C=0.
    - Si es 2D se devuelve tal cual.
    - Otros casos -> ValueError.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, 0]
    raise ValueError(f"Dimensiones no soportadas para imagen: {arr.shape}")


def build_render_callback(
    endpoint: str,
    get_client_func: Callable[[], MitsubaAPIClient],
    select_first_band: bool = True,
) -> Callable[[], Tuple[Optional[Image.Image], Optional[str]]]:
    """Crea un callback estándar para un endpoint de render que devuelve .npy crudo.

    Args:
        endpoint: Ruta relativa después de base_url (ej. /render/air/blackbody)
        get_client_func: función para obtener el cliente (inyectable)
        select_first_band: si True y array es 3D toma la primera banda

    Returns:
        Función sin argumentos que retorna (PIL.Image | None, ruta_archivo | None)
    """

    def _callback():  # type: ignore
        client = get_client_instance(get_client_func)
        try:
            r = client.session.get(f"{client.base_url}{endpoint}")
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tf:
                tf.write(r.content)
                arr = np.load(tf.name)
                if select_first_band:
                    try:
                        arr2d = pick_first_band_if_needed(arr)
                    except ValueError:
                        # fallback: colapsar todo a 2D vía media
                        arr2d = np.nanmean(arr, axis=-1) if arr.ndim == 3 else arr
                else:
                    # Para mapas que ya son 2D, o se quiere usar íntegro
                    arr2d = arr if arr.ndim == 2 else pick_first_band_if_needed(arr)
                pil_img = normalize_to_uint8(arr2d)
                return pil_img, tf.name
        except Exception:
            return None, None

    return _callback

__all__ = [
    "normalize_to_uint8",
    "pick_first_band_if_needed",
    "build_render_callback",
]
