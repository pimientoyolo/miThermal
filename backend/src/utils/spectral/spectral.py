"""
Utilidades para el procesamiento de datos espectrales.

Este módulo contiene funciones para cargar, procesar e interpolar datos espectrales,
específicamente para trabajar con respuestas espectrales de cámaras FLIR.
"""

import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, Union

def load_spectral_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga datos espectrales desde un archivo CSV.

    Args:
        csv_path (str): Ruta al archivo CSV con datos espectrales.
            Se espera que el CSV tenga columnas 'x' e 'y'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays de longitudes de onda (x) y respuesta espectral (y)
    """
    try:
        # Verificar si el archivo existe
        if not os.path.isfile(csv_path):
            print(f"El archivo no existe: {csv_path}")
            return np.array([]), np.array([])
            
        # Leer el archivo CSV usando pandas
        df = pd.read_csv(csv_path, delimiter=',', skipinitialspace=True)
        
        # Extraer columnas x e y
        x_data = df['x'].values
        y_data = df['y'].values
        
        return x_data, y_data
    
    except Exception as e:
        print(f"Error al cargar los datos espectrales: {e}")
        return np.array([]), np.array([])

def interpolate_spectral_response(
    wavelengths: np.ndarray, 
    response: np.ndarray, 
    new_wavelengths: Union[np.ndarray, list],
    method: str = 'cubic'
) -> np.ndarray:
    """
    Interpola la respuesta espectral usando interpolación cúbica.

    Args:
        wavelengths (np.ndarray): Array de longitudes de onda originales.
        response (np.ndarray): Array de valores de respuesta originales.
        new_wavelengths (np.ndarray): Nuevas longitudes de onda para interpolar.
        method (str, opcional): Método de interpolación ('linear', 'cubic', 'quadratic'). 
                               Por defecto es 'cubic'.

    Returns:
        np.ndarray: Valores de respuesta interpolados para las nuevas longitudes de onda.
    """
    # Asegurarse de que new_wavelengths sea un array de numpy
    if not isinstance(new_wavelengths, np.ndarray):
        new_wavelengths = np.array(new_wavelengths)
    
    # Eliminar valores duplicados en las longitudes de onda originales
    unique_indices = np.unique(wavelengths, return_index=True)[1]
    unique_wavelengths = wavelengths[unique_indices]
    unique_response = response[unique_indices]
    
    if len(unique_indices) < len(wavelengths):
        print(f"Advertencia: Se eliminaron {len(wavelengths) - len(unique_indices)} valores duplicados de longitud de onda.")
        
    if len(unique_wavelengths) < 4 and method == 'cubic':
        print("Advertencia: Se necesitan al menos 4 puntos para interpolación cúbica.")
        print("Cambiando a interpolación lineal...")
        method = 'linear'
    
    # Crear el interpolador
    f = interpolate.interp1d(
        unique_wavelengths, 
        unique_response, 
        kind=method, 
        bounds_error=False,
        fill_value=(unique_response[0], unique_response[-1])  # Extrapola con los valores extremos
    )
    
    # Realizar la interpolación
    interpolated_response = f(new_wavelengths)
    
    return interpolated_response

def get_absolute_path(relative_path: str) -> str:
    """
    Convierte una ruta relativa en una ruta absoluta basada en la ubicación del script actual.
    
    Args:
        relative_path (str): Ruta relativa al archivo
        
    Returns:
        str: Ruta absoluta al archivo
    """
    # Obtener el directorio base donde se encuentra este script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Si la ruta ya comienza con '/', asumimos que es absoluta
    if os.path.isabs(relative_path):
        return relative_path
    
    # Eliminar '../' inicial si existe, ya que base_dir ya está un nivel arriba
    if relative_path.startswith("../"):
        relative_path = relative_path[3:]
    
    # Construir la ruta absoluta
    abs_path = os.path.join(base_dir, relative_path)
    
    return abs_path

def interpolate_flir_spectral_response(
    csv_path: str = "raw/plot-data.csv", 
    new_wavelengths: Optional[Union[np.ndarray, list]] = None,
    num_points: int = 1000,
    method: str = 'cubic'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga e interpola la respuesta espectral de una cámara FLIR.

    Args:
        csv_path (str, opcional): Ruta al archivo CSV con datos espectrales (relativa a la raíz del proyecto).
        new_wavelengths (np.ndarray, opcional): Longitudes de onda específicas para interpolar.
                                              Si es None, genera un array lineal.
        num_points (int, opcional): Número de puntos para la interpolación si new_wavelengths es None.
        method (str, opcional): Método de interpolación. Por defecto es 'cubic'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tupla con (wavelengths, response) interpolados.
    """
    # Convertir a ruta absoluta
    abs_csv_path = get_absolute_path(csv_path)
    
    # Cargar datos originales
    wavelengths, response = load_spectral_data(abs_csv_path)
    
    if len(wavelengths) == 0:
        raise ValueError(f"No se pudieron cargar datos desde {abs_csv_path}")
    
    # Si no se proporcionan nuevas longitudes de onda, crear un array lineal
    if new_wavelengths is None:
        new_wavelengths = np.linspace(
            wavelengths.min(), 
            wavelengths.max(), 
            num_points
        )
    
    # Interpolar respuesta
    interpolated_response = interpolate_spectral_response(
        wavelengths, 
        response, 
        new_wavelengths,
        method
    )
    
    return new_wavelengths, interpolated_response

def plot_spectral_response(
    wavelengths: np.ndarray, 
    response: np.ndarray, 
    original_wavelengths: Optional[np.ndarray] = None,
    original_response: Optional[np.ndarray] = None,
    title: str = "Respuesta Espectral",
    x_label: str = "Longitud de onda (µm)",
    y_label: str = "Respuesta normalizada",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza la respuesta espectral y opcionalmente los datos originales.

    Args:
        wavelengths (np.ndarray): Longitudes de onda interpoladas.
        response (np.ndarray): Valores de respuesta interpolados.
        original_wavelengths (np.ndarray, opcional): Longitudes de onda originales.
        original_response (np.ndarray, opcional): Respuesta original.
        title (str, opcional): Título del gráfico.
        x_label (str, opcional): Etiqueta del eje x.
        y_label (str, opcional): Etiqueta del eje y.
        save_path (str, opcional): Ruta para guardar la figura.
    """
    plt.figure(figsize=(10, 6))
    
    # Graficar la respuesta interpolada
    plt.plot(wavelengths, response, '-', label="Interpolación", color='blue', linewidth=2)
    
    # Si se proporcionan los datos originales, graficarlos también
    if original_wavelengths is not None and original_response is not None:
        plt.scatter(original_wavelengths, original_response, 
                   label="Datos originales", color='red', s=30)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Guardar la figura si se proporciona una ruta
    if save_path:
        # Crear el directorio de salida si no existe
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Se ha creado el directorio: {output_dir}")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()

def example_usage():
    """Función de ejemplo que muestra cómo usar las utilidades."""
    # Ruta al archivo CSV - usar ruta relativa a la raíz del proyecto
    csv_path = "raw/plot-data.csv"
    
    # Obtener ruta absoluta para cargar los datos originales
    abs_path = get_absolute_path(csv_path)
    original_wavelengths, original_response = load_spectral_data(abs_path)
    
    # Interpolar respuesta espectral usando todas las longitudes de onda disponibles
    # No especificamos new_wavelengths para usar el rango completo de los datos originales
    wavelengths, interpolated_response = interpolate_flir_spectral_response(
        csv_path, 
        num_points=2000  # Mayor resolución para obtener una curva más suave
    )
    
    # Visualizar resultados
    plot_spectral_response(
        wavelengths,
        interpolated_response,
        original_wavelengths,
        original_response,
        title="Respuesta Espectral Interpolada - Cámara FLIR",
        save_path="output/spectral_response.png"
    )
    
    print(f"Datos originales: {len(original_wavelengths)} puntos")
    print(f"Datos interpolados: {len(interpolated_response)} puntos")

if __name__ == "__main__":
    example_usage()
