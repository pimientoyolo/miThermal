"""Utilidades de objetos 3D"""

from .objects import ObjectUtils
from .properties import SpectralProperty, ObjectProperties
from .object_db import ObjectDatabase, get_object_db
from .family_manager import FamilyManager, ObjectFamily

__all__ = [
    'ObjectUtils',
    # Nueva arquitectura para propiedades
    'SpectralProperty',
    'ObjectProperties',
    'ObjectDatabase',
    'get_object_db',
    # Gestión de familias
    'FamilyManager',
    'ObjectFamily',
]
