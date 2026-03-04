"""
Funciones de transformación geométrica y matrices de rotación.
"""

import numpy as np


def degrees_to_radians(degrees: float) -> float:
    """Convierte grados a radianes."""
    return np.deg2rad(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convierte radianes a grados."""
    return np.rad2deg(radians)


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """
    Crea matriz de rotación alrededor del eje X.
    
    Args:
        angle_deg: Ángulo en grados
        
    Returns:
        Matriz de rotación 3x3
    """
    r = np.deg2rad(angle_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """
    Crea matriz de rotación alrededor del eje Y.
    
    Args:
        angle_deg: Ángulo en grados
        
    Returns:
        Matriz de rotación 3x3
    """
    r = np.deg2rad(angle_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """
    Crea matriz de rotación alrededor del eje Z.
    
    Args:
        angle_deg: Ángulo en grados
        
    Returns:
        Matriz de rotación 3x3
    """
    r = np.deg2rad(angle_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def euler_to_rotation_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """
    Convierte ángulos de Euler (XYZ intrínseco) a matriz de rotación.
    Patrón: R = Rz(z) @ Ry(y) @ Rx(x)
    
    Args:
        x_deg: Rotación alrededor del eje X en grados
        y_deg: Rotación alrededor del eje Y en grados
        z_deg: Rotación alrededor del eje Z en grados
        
    Returns:
        Matriz de rotación 3x3 compuesta
    """
    rx = rotation_matrix_x(x_deg)
    ry = rotation_matrix_y(y_deg)
    rz = rotation_matrix_z(z_deg)
    return rz @ ry @ rx


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Convierte matriz de rotación a ángulos de Euler (XYZ intrínseco).
    Extrae los ángulos de rotación de una matriz.
    
    Args:
        rotation_matrix: Matriz de rotación 3x3
        
    Returns:
        Tupla (angle_x, angle_y, angle_z) en grados
    """
    # Extraer ángulos de la matriz
    R = rotation_matrix
    
    # Verificar singularidad
    sin_y = np.clip(R[2, 0], -1, 1)
    
    if abs(sin_y - 1.0) < 1e-6:
        # Gimbal lock: pitch = 90 degrees
        angle_x = np.arctan2(-R[0, 1], R[1, 1])
        angle_y = np.pi / 2
        angle_z = 0
    elif abs(sin_y + 1.0) < 1e-6:
        # Gimbal lock: pitch = -90 degrees
        angle_x = np.arctan2(R[0, 1], R[1, 1])
        angle_y = -np.pi / 2
        angle_z = 0
    else:
        angle_y = np.arcsin(sin_y)
        angle_x = np.arctan2(-R[2, 1], R[2, 2])
        angle_z = np.arctan2(-R[1, 0], R[0, 0])
    
    return (np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z))


def clamp_value(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Limita un valor entre mínimo y máximo.
    
    Args:
        value: Valor a limitar
        min_val: Valor mínimo
        max_val: Valor máximo
        
    Returns:
        Valor limitado al rango [min_val, max_val]
    """
    return max(min_val, min(max_val, value))
