"""
Funciones para la creación y configuración de escenas en Mitsuba.
"""
import mitsuba as mi
from mitsuba import ScalarTransform4f as T
from .visualization_utils import lista_a_string

def create_dragon_scene(wave_lengths, emision, material_name='stone'):
    """
    Crea una escena con un modelo de dragón con emisión de área.
    
    Args:
        wave_lengths (np.ndarray): Longitudes de onda para la emisión
        emision (np.ndarray): Valores de emisión correspondientes
        material_name (str): Nombre del material (para referencia)
        
    Returns:
        mi.Scene: Escena con el dragón emisor
    """
    # Definir el dragón con la emisión específica
    dragon = {
        'type': 'obj',
        'filename': 'mitsuba/scenes/objects/Dragon.obj',
        'to_world': T().translate([0, 0, 0]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.1, 0.2, 0.3]
            }
        },
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'irregular',
                'wavelengths': lista_a_string(wave_lengths),
                'values': lista_a_string(emision),
            }
        }
    }

    # Crear la escena con el dragón
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
        },
        'dragon': dragon,
    })

    return scene

def create_point_source_scene(spd_wavelengths, spd_values, sphere_radius=1.0):
    """
    Crea una escena con un emisor puntual y una esfera de referencia.
    
    Args:
        spd_wavelengths (list): Longitudes de onda del emisor
        spd_values (list): Valores de emisión correspondientes
        sphere_radius (float): Radio de la esfera de referencia
        
    Returns:
        mi.Scene: Escena con el emisor puntual
    """
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
        },
        # Emisor puntual usando los datos espectrales
        'point_light': {
            'type': 'point',
            'position': [0, 0, 0],  # ubicado en el origen
            'intensity': {
                'type': 'irregular',
                'wavelengths': lista_a_string(spd_wavelengths),
                'values': lista_a_string(spd_values)
            }
        },
        # Añadir una esfera para visualizar dónde está el emisor puntual
        'reference_sphere': {
            'type': 'sphere',
            'center': [0, 0, 0],   # Ubicada en el origen
            'radius': sphere_radius,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [1.0, 0.8, 0.2]  # Color amarillento
                }
            }
        }
    })

    return scene

def create_area_emitter_scene(spd_wavelengths, spd_values, sphere_radius=5.0):
    """
    Crea una escena con una esfera emisora (emisor de área).
    
    Args:
        spd_wavelengths (list): Longitudes de onda del emisor
        spd_values (list): Valores de emisión correspondientes
        sphere_radius (float): Radio de la esfera emisora
        
    Returns:
        mi.Scene: Escena con la esfera emisora
    """
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
        },
        # Esfera con emisor de área que usa los datos espectrales
        'emissive_sphere': {
            'type': 'sphere',
            'center': [0, 0, 0],  # Ubicada en el origen
            'radius': sphere_radius,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [1.0, 0.8, 0.2]  # Color amarillento
                }
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'irregular',
                    'wavelengths': lista_a_string(spd_wavelengths),
                    'values': lista_a_string(spd_values)
                }
            }
        }
    })

    return scene

def create_multi_sphere_scene(spd_wavelengths, spd_values, base_distance=10.0, sphere_radius=2.0, 
                              distance_factors=[1, 2, 4, 8]):
    """
    Crea una escena con múltiples esferas emisoras a diferentes distancias.
    
    Args:
        spd_wavelengths (list): Longitudes de onda del emisor
        spd_values (list): Valores de emisión correspondientes
        base_distance (float): Distancia base para el primer factor
        sphere_radius (float): Radio de las esferas emisoras
        distance_factors (list): Factores de distancia para las esferas
        
    Returns:
        mi.Scene: Escena con múltiples esferas emisoras
    """
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
        }
    }
    
    # Crear cada esfera a su distancia correspondiente
    for i, factor in enumerate(distance_factors):
        distance = base_distance * factor
        
        # Posición de la esfera: en línea a lo largo del eje Z positivo
        position = [0, 0, distance]
        
        sphere_name = f'sphere_{factor}x'
        scene_dict[sphere_name] = {
            'type': 'sphere',
            'center': position,
            'radius': sphere_radius,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.9, 0.7, 0.3]  # Color dorado
                }
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'irregular',
                    'wavelengths': lista_a_string(spd_wavelengths),
                    'values': lista_a_string(spd_values)
                }
            }
        }
    
    scene = mi.load_dict(scene_dict)
    return scene
