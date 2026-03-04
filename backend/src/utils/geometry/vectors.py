"""
Funciones de operaciones con vectores 3D.
"""

import numpy as np


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector 3D.
    
    Args:
        vec: Vector 3D
        
    Returns:
        Vector normalizado (magnitud = 1)
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Producto cruz entre dos vectores 3D.
    
    Args:
        a: Primer vector
        b: Segundo vector
        
    Returns:
        Vector perpendicular a ambos
    """
    return np.cross(a, b)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Producto punto entre dos vectores.
    
    Args:
        a: Primer vector
        b: Segundo vector
        
    Returns:
        Escalar resultante del producto punto
    """
    return np.dot(a, b)


def vector_length(vec: np.ndarray) -> float:
    """
    Calcula la longitud (magnitud) de un vector.
    
    Args:
        vec: Vector
        
    Returns:
        Longitud del vector
    """
    return np.linalg.norm(vec)


def vector_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.
    
    Args:
        a: Primer punto
        b: Segundo punto
        
    Returns:
        Distancia entre los puntos
    """
    return np.linalg.norm(b - a)


def lerp_vector(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolación lineal entre dos vectores.
    
    Args:
        a: Vector inicial
        b: Vector final
        t: Factor de interpolación [0, 1]
        
    Returns:
        Vector interpolado
    """
    return a * (1.0 - t) + b * t
