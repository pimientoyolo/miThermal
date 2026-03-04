"""
Módulo de cache para optimizar operaciones costosas.
"""

from .emission_cache import EmissionCacheManager, get_cache_manager
from .emissivity_index import EmissivityFileIndex, get_emissivity_index

__all__ = [
    "EmissionCacheManager", 
    "get_cache_manager",
    "EmissivityFileIndex",
    "get_emissivity_index"
]
