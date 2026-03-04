"""Utilidades de geometría y transformaciones"""

from .transformations import (
    degrees_to_radians,
    radians_to_degrees,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    clamp_value,
)

from .vectors import (
    normalize_vector,
    cross_product,
    dot_product,
    vector_length,
    vector_distance,
    lerp_vector,
)

__all__ = [
    # Transformations
    'degrees_to_radians',
    'radians_to_degrees',
    'rotation_matrix_x',
    'rotation_matrix_y',
    'rotation_matrix_z',
    'euler_to_rotation_matrix',
    'rotation_matrix_to_euler',
    'clamp_value',
    # Vectors
    'normalize_vector',
    'cross_product',
    'dot_product',
    'vector_length',
    'vector_distance',
    'lerp_vector',
]
