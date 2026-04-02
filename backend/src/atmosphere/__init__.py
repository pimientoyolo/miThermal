"""
Modelos y cálculos atmosféricos
"""

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
    # Medium
    'create_homogeneous_medium',
    'lista_a_string',
    
    # Gas Manager (new system)
    'AtmosphericGasManager',
    'get_gas_manager',
    'AVAILABLE_GASES',
]
