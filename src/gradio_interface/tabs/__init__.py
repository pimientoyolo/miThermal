"""
Módulos de pestañas para la interfaz Gradio
"""

from .simulation import create_simulation_tab
from .spectral_analysis import create_spectral_analysis_tab
from .advanced_config import create_advanced_config_tab
from .export import create_export_tab

__all__ = [
    'create_simulation_tab',
    'create_spectral_analysis_tab', 
    'create_advanced_config_tab',
    'create_export_tab'
]
