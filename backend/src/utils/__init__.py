"""
Utilidades generales del proyecto
"""

from src.utils.decorators import (
    validate_file_exists,
    validate_directory_exists,
    validate_file_extension,
    log_execution,
    handle_file_errors,
    require_scene_loaded,
)

from src.utils.helpers import (
    # Transformaciones
    degrees_to_radians,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    clamp_value,
    # Archivos
    find_files_by_pattern,
    ensure_directory_exists,
    clean_directory,
    get_relative_path,
    normalize_path,
    # Datos
    load_numpy_array,
    save_numpy_array,
    # Wavelengths
    generate_wavelength_range,
    wavelengths_nm_to_um,
    wavelengths_um_to_nm,
)

__all__ = [
    # Decoradores
    "validate_file_exists",
    "validate_directory_exists",
    "validate_file_extension",
    "log_execution",
    "handle_file_errors",
    "require_scene_loaded",
    # Helpers
    "degrees_to_radians",
    "rotation_matrix_x",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "clamp_value",
    "find_files_by_pattern",
    "ensure_directory_exists",
    "clean_directory",
    "get_relative_path",
    "normalize_path",
    "load_numpy_array",
    "save_numpy_array",
    "generate_wavelength_range",
    "wavelengths_nm_to_um",
    "wavelengths_um_to_nm",
]
