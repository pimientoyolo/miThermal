"""
Modelos y cálculos atmosféricos
"""

from .attenuation import (
    get_attenuation,
    read_air_attenuation_file,
    ATMOSPHERIC_GAS_FILES,
)

from .medium import (
    create_homogeneous_medium,
    lista_a_string,
)

from .gas_manager import (
    AtmosphericGasManager,
    get_gas_manager,
    AVAILABLE_GASES,
)

__all__ = [
    # Legacy attenuation functions - will be deprecated
    'get_attenuation',
    'read_air_attenuation_file',
    'ATMOSPHERIC_GAS_FILES',
    
    # Medium
    'create_homogeneous_medium',
    'lista_a_string',
    
    # Gas Manager (new system)
    'AtmosphericGasManager',
    'get_gas_manager',
    'AVAILABLE_GASES',
]
