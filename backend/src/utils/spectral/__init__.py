"""Utilidades espectrales - Procesamiento de datos y emisiones espectrales"""

from .emission import blackbody_radiance_nm, load_material_data, get_material_signature, load_spd_data
from .spectral import (
    load_spectral_data,
    interpolate_spectral_response,
    get_absolute_path,
    interpolate_flir_spectral_response,
    plot_spectral_response
)

__all__ = [
    'blackbody_radiance_nm',
    'load_material_data',
    'get_material_signature',
    'load_spd_data',
    'load_spectral_data',
    'interpolate_spectral_response',
    'get_absolute_path',
    'interpolate_flir_spectral_response',
    'plot_spectral_response'
]
