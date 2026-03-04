"""
Utilidades para la creación y configuración de sensores espectrales.

Este módulo contiene funciones para crear y configurar sensores en Mitsuba con
respuestas espectrales gaussianas, así como para ajustar el campo de visión (FOV)
según la distancia para mantener un ángulo sólido constante.
"""
import math
import numpy as np
#from .visualization_utils import lista_a_string

def gausian(lambdas: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the Gaussian function value at x with mean mu and standard deviation sigma.
    Args:
        lambdas (np.ndarray): The input values (wavelengths).
        mu (float): The mean of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.
    """
    return np.exp(-0.5 * ((lambdas - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def camera_response_gaussian(wavelength: int, sigma: float, k: float = 3, n_wavelents: int = 5):
    """
    Camera response function

    Args:
        wavelength (int): Wavelength in nm.
        sigma (float): Standard deviation of the Gaussian.
        k (float, optional): Number of standard deviations. Default is 3.
        n_wavelents (int, optional): Number of wavelengths. Default is 5.
    """
    # define the wavelength range (n values)
    wavelengths = np.linspace(wavelength - k * sigma, wavelength + k * sigma, n_wavelents)

    # define the Gaussian function
    gaussian_values = gausian(wavelengths, wavelength, sigma)

    # normalize the Gaussian function (0,1)
    gaussian_values = (gaussian_values - np.min(gaussian_values)) / (np.max(gaussian_values) - np.min(gaussian_values))

    return wavelengths, gaussian_values

def create_specfilm(num_bands: int, w_min: int, w_max: int) -> dict:
    """
    Crea solo el film para un sensor espectral, espaciando bandas entre w_min y w_max.
    Todas las bandas tienen valores de 1.0 y mantienen la estructura para Mitsuba/XML.

    Args:
        num_bands (int): Número de bandas espectrales.
        w_min (int): Longitud de onda mínima.
        w_max (int): Longitud de onda máxima.

    Returns:
        dict: Diccionario film listo para Mitsuba/XML.
    """
    film_dic = {}
    # Espaciado uniforme de bandas
    bands = np.linspace(w_min, w_max, num_bands).astype(int)
    distancia = bands[1] - bands[0] if num_bands > 1 else 1
    separar = distancia // 2
    for wave_length in bands:
        wmin_band = int(wave_length - separar)
        wmax_band = int(wave_length + separar)
        film_dic[f"band_{wave_length}"] = {
            "@type": "regular",
            "@wavelength_min": wmin_band,
            "@wavelength_max": wmax_band,
            "@values": "1.0, 1.0"
        }
    return film_dic

# def load_sensor_gaussian(r: float, phi: float, theta: float, wave_lengths: np.ndarray, sigma: float, 
#                         base_distance: float = None, base_fov: float = 40, k: float = 3, 
#                         n_wavelents: int = 5, rfilter_type: str = 'tent'):
#     """
#     Load a sensor with Gaussian response, adjusting FOV based on distance to maintain constant solid angle.

#     Args:
#         r (float): Radius of the sensor.
#         phi (float): Azimuthal angle in degrees.
#         theta (float): Polar angle in degrees.
#         wave_lengths (np.ndarray): Wavelengths for the Gaussian response.
#         sigma (float): Standard deviation of the Gaussian.
#         base_distance (float, optional): Reference distance for FOV calculation. If None, uses fixed FOV.
#         base_fov (float, optional): Base FOV at reference distance. Default is 40 degrees.
#         k (float, optional): Number of standard deviations. Default is 3.
#         n_wavelents (int, optional): Number of wavelengths. Default is 5.
#         rfilter_type (str, optional): Type of reconstruction filter. Default is 'tent'.
#     """
#     z = r * np.sin(math.radians(theta))
#     y = r * np.cos(math.radians(theta)) * np.sin(math.radians(phi))
#     x = r * np.cos(math.radians(theta)) * np.cos(math.radians(phi))
    
#     # Ajustar FOV según la distancia para mantener el ángulo sólido constante
#     fov = base_fov
#     if base_distance is not None and r != base_distance:
#         # Usamos la función auxiliar para calcular el FOV ajustado
#         fov = calculate_adjusted_fov(base_fov, base_distance, r)
#         print(f"  Ajuste de FOV: {fov:.2f}° a distancia {r}")
    
#     origin = np.array([x, y, z])
#     film_dic = {
#         'type': 'specfilm',
#         'width': 256,
#         'height': 256,
#         'rfilter': {
#             'type': rfilter_type,
#         },
#         'sample_border': True,
#         'compensate': True
#     }

#     # create sensor uniform
#     distancia = wave_lengths[1] - wave_lengths[0]
#     separar = distancia // 2

#     for wave_length in wave_lengths:
#         w_min = int(wave_length - separar)
#         w_max = int(wave_length + separar)

#         values = '0.1, 1.0'

#         film_dic[f'band_{wave_length}'] = {
#             # 'type': 'uniform',
#             # 'wavelength_min': w_min,
#             # 'wavelength_max': w_max,
#             # 'value' : 1
#             'type': 'regular',
#             'wavelength_min': w_min,
#             'wavelength_max': w_max,
#             'values': values,
#         }

#     return mi.load_dict({
#         'type': 'perspective',
#         'fov': fov,  # Usando el FOV ajustado
#         'to_world': T().look_at(
#             origin=origin,
#             target=[0, 0, 0],
#             up=[0, 0, 1]
#         ),
#         'sampler': {
#             'type': 'independent',
#             'sample_count': 1
#         },
#         'film': film_dic,
#     })

def calculate_adjusted_fov(base_fov: float, base_distance: float, current_distance: float) -> float:
    """
    Calcula el FOV ajustado para mantener un ángulo sólido constante a diferentes distancias.
    
    Args:
        base_fov (float): FOV base en la distancia de referencia
        base_distance (float): Distancia de referencia donde se definió el FOV base
        current_distance (float): Distancia actual a la que se necesita ajustar el FOV
        
    Returns:
        float: FOV ajustado en grados
    """
    return 2 * math.degrees(math.atan(math.tan(math.radians(base_fov/2)) * (base_distance / current_distance)))

def lista_a_string(lista):
    """
    Convierte una lista de números en una cadena formateada para Mitsuba.

    Args:
        lista (list): La lista de números a convertir.

    Returns:
        str: La cadena formateada.
    """
    return ', '.join(f'{num:.6f}' for num in lista)

# def load_fixed_sensor_gaussian(x: float, y: float, z: float, target_x: float = 0, target_y: float = 0, target_z: float = 0,
#                               wave_lengths: np.ndarray = None, sigma: float = 10, fov: float = 60,
#                               k: float = 3, n_wavelents: int = 5, rfilter_type: str = 'tent'):
#     """
#     Crea un sensor con respuesta gaussiana en una posición fija mirando hacia un objetivo.

#     Args:
#         x, y, z (float): Posición de la cámara
#         target_x, target_y, target_z (float): Punto hacia donde mira la cámara
#         wave_lengths (np.ndarray): Longitudes de onda para la respuesta gaussiana
#         sigma (float): Desviación estándar de la gaussiana
#         fov (float): Campo de visión en grados
#         k (float): Número de desviaciones estándar
#         n_wavelents (int): Número de longitudes de onda
#         rfilter_type (str): Tipo de filtro de reconstrucción
#     """
#     if wave_lengths is None:
#         wave_lengths = np.linspace(380, 780, 41).astype(int)
    
#     origin = np.array([x, y, z])
#     target = np.array([target_x, target_y, target_z])
    
#     # Calcular la dirección de vista
#     look_direction = target - origin
#     look_direction = look_direction / np.linalg.norm(look_direction)
    
#     # Vector up por defecto
#     up = np.array([0, 1, 0])
    
#     film_dic = {
#         'type': 'specfilm',
#         'width': 256,
#         'height': 256,
#         'rfilter': {
#             'type': rfilter_type,
#         },
#         'sample_border': True,
#     }

#     # Crear diccionario de respuesta espectral y añadirlo directamente al film
#     for i, wavelength in enumerate(wave_lengths):
#         wavelengths, gaussian_values = camera_response_gaussian(wavelength, sigma, k, n_wavelents)
        
#         film_dic[f'response_{i}'] = {
#             'type': 'irregular',
#             'wavelengths': lista_a_string(wavelengths),
#             'values': lista_a_string(gaussian_values)
#         }

#     # Configurar la transformación de la cámara
#     transform = T().look_at(
#         origin=origin,
#         target=target,
#         up=up
#     )

#     sensor_dict = {
#         'type': 'perspective',
#         'fov': fov,
#         'to_world': transform,
#         'film': film_dic
#     }
    
#     print(f"Sensor fijo creado en posición ({x}, {y}, {z}) mirando hacia ({target_x}, {target_y}, {target_z})")
#     print(f"FOV: {fov}°")
    
#     return mi.load_dict(sensor_dict)

def create_specfilm_bands(wavelengths: np.ndarray) -> list:
    """
    Crea las bandas para el film de un sensor espectral usando un array de longitudes de onda.
    Cada longitud de onda se usa como centro y se extiende igualmente hacia ambos lados.

    Args:
        wavelengths (np.ndarray): Array con las longitudes de onda centrales para cada banda.

    Returns:
        list: Lista con las bandas para el film del sensor espectral.
    """
    band_list = []

    for i, wave_length in enumerate(wavelengths):

        if i == 0:  # Primera banda
            wmin_band = int(wave_length)
            wmax_band = int(wave_length+1)
        else:    
            wmin_band = int(wave_length-1)
            wmax_band = int(wave_length)

        band_list.append({
            "@type": "regular",
            "@name": f"band_{int(wave_length)}",
            "string": [
                {"@name": "values", "@value": "1.0, 1.0"},
            ],
            "float":[
                {"@name": "wavelength_min", "@value": wmin_band},
                {"@name": "wavelength_max", "@value": wmax_band},
            ]
        })
    return band_list
