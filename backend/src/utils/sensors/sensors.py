"""
Utilidades para la creación y configuración de sensores espectrales.

Este módulo contiene funciones para crear y configurar sensores en Mitsuba con
respuestas espectrales gaussianas, así como para ajustar el campo de visión (FOV)
según la distancia para mantener un ángulo sólido constante.
"""
import math
import numpy as np

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

def create_specfilm_bands(wavelengths: np.ndarray) -> list:
    """
    Crea las bandas para el plugin 'specfilm' de Mitsuba.
    Genera intervalos contiguos centrados en cada longitud de onda.

    Args:
        wavelengths (np.ndarray): Longitudes de onda centrales en nm.

    Returns:
        list: Lista de diccionarios <spectrum> para el film.
    """
    band_list = []
    num_bands = len(wavelengths)
    
    if num_bands == 0:
        return []
        
    # Calcular el ancho de banda (distancia entre centros)
    if num_bands > 1:
        delta = float(wavelengths[1] - wavelengths[0])
    else:
        delta = 10.0
        
    half_delta = delta / 2.0

    for wave_length in wavelengths:
        wmin = float(wave_length - half_delta)
        wmax = float(wave_length + half_delta)

        # Mitsuba specfilm: cada <spectrum> define una SRF (Sensor Response Function)
        # que se convierte en un canal del archivo EXR/Tensor de salida.
        band_list.append({
            "@type": "regular",
            "@name": f"band_{wave_length:.2f}",
            "string": {"@name": "values", "@value": "1.0, 1.0"},
            "float": [
                {"@name": "wavelength_min", "@value": f"{wmin:.4f}"},
                {"@name": "wavelength_max", "@value": f"{wmax:.4f}"},
            ]
        })
    return band_list
