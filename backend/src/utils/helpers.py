"""
Utilidades y helpers comunes.
Centraliza funciones que se repiten en múltiples archivos.

NOTA: Las funciones de transformación geométrica se movieron a utils.geometry
      Las funciones de atmósfera se movieron a atmosphere
"""

import os
import glob
from pathlib import Path
from typing import Optional, List
import numpy as np

# Retrocompatibilidad: importar funciones de geometría desde su nueva ubicación
from .geometry import (
    degrees_to_radians,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    clamp_value,
)


# ============================================================================
# Utilidades de archivos y directorios
# ============================================================================

def find_files_by_pattern(directory: str, pattern: str = "*") -> List[str]:
    """
    Busca archivos en un directorio por patrón.
    
    Args:
        directory: Directorio a buscar
        pattern: Patrón glob (ej: "*.txt", "scene_*.xml")
    
    Returns:
        Lista de nombres de archivo (sin ruta)
    """
    path = Path(directory)
    if not path.is_dir():
        return []
    
    files = [
        os.path.basename(p) 
        for p in glob.glob(os.path.join(directory, pattern)) 
        if os.path.isfile(p)
    ]
    return sorted(files, key=str.lower)


def ensure_directory_exists(dir_path: str) -> None:
    """Crea un directorio si no existe."""
    os.makedirs(dir_path, exist_ok=True)


def clean_directory(dir_path: str, keep_subdirs: bool = False) -> None:
    """
    Limpia un directorio eliminando su contenido.
    
    Args:
        dir_path: Ruta del directorio a limpiar
        keep_subdirs: Si False, elimina también subdirectorios
    """
    import shutil
    
    if not os.path.exists(dir_path):
        return
    
    for entry in os.listdir(dir_path):
        entry_path = os.path.join(dir_path, entry)
        try:
            if os.path.isdir(entry_path) and not keep_subdirs:
                shutil.rmtree(entry_path)
            elif os.path.isfile(entry_path):
                os.remove(entry_path)
        except OSError:
            pass  # Continuar si hay error


def get_relative_path(absolute_path: str, base_path: Optional[str] = None) -> str:
    """
    Convierte ruta absoluta a relativa.
    Maneja diferencias entre sistemas (Windows vs Unix).
    """
    obj_path = Path(absolute_path)
    if obj_path.is_absolute():
        obj_path = obj_path.relative_to(obj_path.anchor)
    return str(obj_path)


def normalize_path(path: str) -> str:
    """Normaliza una ruta (cross-platform)."""
    return str(Path(path))


# ============================================================================
# Utilidades de datos
# ============================================================================

def load_numpy_array(file_path: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Carga un archivo .npy de forma segura.
    
    Args:
        file_path: Ruta del archivo
        default: Valor por defecto si hay error
    
    Returns:
        Array o default si hay error
    """
    try:
        if os.path.exists(file_path):
            return np.load(file_path)
        return default
    except Exception:
        return default


def save_numpy_array(file_path: str, data: np.ndarray) -> bool:
    """
    Guarda un array numpy de forma segura.
    
    Returns:
        True si tuvo éxito, False en caso contrario
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, data)
        return True
    except Exception:
        return False


# ============================================================================
# Utilidades de wavelengths y bandas espectrales
# ============================================================================

def generate_wavelength_range(start_nm: int, end_nm: int, num_bands: int) -> np.ndarray:
    """
    Genera array de longitudes de onda.
    
    Args:
        start_nm: Longitud de onda inicial (nm)
        end_nm: Longitud de onda final (nm)
        num_bands: Número de bandas
    
    Returns:
        Array de longitudes de onda en nm
    """
    return np.linspace(start_nm, end_nm, num_bands, endpoint=True, dtype=int)


def wavelengths_nm_to_um(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Convierte longitudes de onda de nm a µm."""
    return wavelengths_nm / 1000.0


def wavelengths_um_to_nm(wavelengths_um: np.ndarray) -> np.ndarray:
    """Convierte longitudes de onda de µm a nm."""
    return wavelengths_um * 1000.0
