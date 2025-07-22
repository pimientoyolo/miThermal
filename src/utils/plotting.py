"""
Funciones de utilidad general para el proyecto de análisis espectral.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def lista_a_string(valores):
    """
    Convierte una lista de valores a una cadena separada por comas.
    
    Args:
        valores (list): Lista de valores a convertir
        
    Returns:
        str: Cadena con valores separados por comas
    """
    return ", ".join(str(v) for v in valores)

def ensure_output_dir(output_dir='output'):
    """
    Asegura que exista un directorio de salida para guardar resultados.
    
    Args:
        output_dir (str): Ruta al directorio de salida
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio '{output_dir}' creado exitosamente.")

def visualize_images(imagenes, distancias, banda=0, title_prefix='', save_path=None):
    """
    Visualiza imágenes renderizadas a diferentes distancias.
    
    Args:
        imagenes (list): Lista de imágenes a visualizar
        distancias (list): Lista de factores de distancia correspondientes
        banda (int): Índice de la banda espectral a visualizar
        title_prefix (str): Prefijo para el título de la gráfica
        save_path (str): Ruta para guardar la gráfica (opcional)
    """
    plt.figure(figsize=(15, 10))
    for i, factor in enumerate(distancias):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imagenes[i][:, :, banda], cmap='plasma')
        plt.colorbar()
        plt.title(f'{title_prefix} Distancia: {factor}x')
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        ensure_output_dir()
        plt.savefig(save_path)
    plt.show()

def analyze_spectral_comparison(pixeles, wave_lengths, spd_wavelengths, spd_values, 
                               distancias, title_suffix='', save_paths=None):
    """
    Realiza un análisis comparativo de los datos espectrales a diferentes distancias.
    
    Args:
        pixeles (list): Lista de valores de píxeles para cada distancia
        wave_lengths (np.ndarray): Longitudes de onda del sensor
        spd_wavelengths (list): Longitudes de onda del emisor
        spd_values (list): Valores de emisión del emisor
        distancias (list): Factores de distancia analizados
        title_suffix (str): Sufijo para el título de las gráficas
        save_paths (dict): Diccionario con rutas para guardar cada gráfica
    """
    ensure_output_dir()
    
    # Comparar las emisiones a diferentes distancias
    plt.figure(figsize=(15, 10))

    # Primero graficar la emisión original del material
    plt.subplot(2, 3, 1)
    plt.plot(spd_wavelengths, spd_values)
    plt.title(f'Emisión original (test.spd)')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Emisión relativa')
    plt.grid(True)

    # Graficar los valores de píxeles para cada distancia
    for i, factor in enumerate(distancias):
        plt.subplot(2, 3, i + 2)
        plt.plot(wave_lengths, pixeles[i])
        plt.title(f'Emisión captada a distancia {factor}x')
        plt.xlabel('Longitud de onda (nm)')
        plt.ylabel('Valor del píxel')
        plt.grid(True)

    plt.tight_layout()
    if save_paths and 'spectra' in save_paths:
        plt.savefig(save_paths['spectra'])
    plt.show()

    # Comparar todas las emisiones en una sola gráfica
    plt.figure(figsize=(12, 8))

    # Normalizar los valores para una mejor comparación
    normalized_pixels = []
    for pixel in pixeles:
        if np.max(pixel) > 0:  # evitar división por cero
            normalized_pixel = pixel / np.max(pixel)
        else:
            normalized_pixel = pixel
        normalized_pixels.append(normalized_pixel)

    # Graficar todas las emisiones normalizadas
    for i, factor in enumerate(distancias):
        plt.plot(wave_lengths, normalized_pixels[i], label=f'Distancia {factor}x')

    plt.title(f'Comparación de emisiones normalizadas a diferentes distancias {title_suffix}')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Emisión normalizada')
    plt.legend()
    plt.grid(True)
    if save_paths and 'comparison' in save_paths:
        plt.savefig(save_paths['comparison'])
    plt.show()

    return normalized_pixels

def analyze_intensity_falloff(pixeles, distancias, title_suffix='', save_paths=None):
    """
    Analiza la caída de intensidad con la distancia y compara con la ley del cuadrado inverso.
    
    Args:
        pixeles (list): Lista de valores de píxeles para cada distancia
        distancias (list): Factores de distancia analizados
        title_suffix (str): Sufijo para el título de las gráficas
        save_paths (dict): Diccionario con rutas para guardar cada gráfica
    
    Returns:
        tuple: (intensidades, intensidad_relativa, error_porcentual)
    """
    ensure_output_dir()
    
    # Comparación de la caída de intensidad con la distancia
    intensidades = [np.max(pixel) for pixel in pixeles]
    intensidad_relativa = [intensidad / intensidades[0] for intensidad in intensidades]

    # La ley del cuadrado inverso predice que la intensidad cae con el cuadrado de la distancia
    ley_cuadrado_inverso = [1 / (factor**2) for factor in distancias]

    plt.figure(figsize=(10, 6))
    plt.plot(distancias, intensidad_relativa, 'o-', label='Intensidad medida')
    plt.plot(distancias, ley_cuadrado_inverso, 's--', label='Ley del cuadrado inverso (1/r²)')
    plt.title(f'Caída de intensidad con la distancia {title_suffix}')
    plt.xlabel('Factor de distancia')
    plt.ylabel('Intensidad relativa')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    if save_paths and 'intensity_falloff' in save_paths:
        plt.savefig(save_paths['intensity_falloff'])
    plt.show()

    # Análisis del error respecto a la ley del cuadrado inverso
    plt.figure(figsize=(10, 6))

    # Calcular el error porcentual respecto a la ley del cuadrado inverso
    error_porcentual = [(medido / teorico - 1) * 100 
                       for medido, teorico in zip(intensidad_relativa, ley_cuadrado_inverso)]

    # Graficar el error porcentual
    plt.bar(range(len(distancias)), error_porcentual)
    plt.xticks(range(len(distancias)), [f'{d}x' for d in distancias])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Error porcentual respecto a la ley del cuadrado inverso')
    plt.xlabel('Factor de distancia')
    plt.ylabel('Error (%)')
    plt.grid(True, axis='y', alpha=0.3)
    if save_paths and 'error' in save_paths:
        plt.savefig(save_paths['error'])
    plt.show()
    
    # Tabla de resultados
    print(f"\nTabla de resultados {title_suffix}")
    print("=" * 80)
    print(f"{'Distancia':^12} | {'Intensidad':^15} | {'Intensidad rel.':^15} | {'Ley 1/r²':^15} | {'Error %':^10}")
    print("-" * 80)
    for i, d in enumerate(distancias):
        print(f"{d:^12.0f}x | {intensidades[i]:^15.8f} | {intensidad_relativa[i]:^15.8f} | {ley_cuadrado_inverso[i]:^15.8f} | {error_porcentual[i]:^10.2f}%")
    print("=" * 80)
    
    return intensidades, intensidad_relativa, error_porcentual

def detailed_comparison(wave_lengths, normalized_pixels, spd_wavelengths, spd_values, distancias, save_path=None):
    """
    Realiza una comparación detallada entre la emisión original y las mediciones a diferentes distancias.
    
    Args:
        wave_lengths (np.ndarray): Longitudes de onda del sensor
        normalized_pixels (list): Lista de valores de píxeles normalizados
        spd_wavelengths (list): Longitudes de onda del emisor
        spd_values (list): Valores de emisión del emisor
        distancias (list): Factores de distancia analizados
        save_path (str): Ruta para guardar la gráfica
        
    Returns:
        list: Error espectral relativo para cada distancia
    """
    ensure_output_dir()
    
    plt.figure(figsize=(15, 12))

    # Interpolar los valores del archivo SPD a las longitudes de onda del sensor
    spd_values_interpolados = np.interp(wave_lengths, spd_wavelengths, spd_values)

    # Normalizar la emisión original para comparación
    spd_norm = spd_values_interpolados / np.max(spd_values_interpolados)
    
    # Lista para almacenar el error relativo para cada distancia
    error_espectral = []

    for i, factor in enumerate(distancias):
        plt.subplot(3, 2, i + 1)
        
        # Graficar la emisión original normalizada
        plt.plot(wave_lengths, spd_norm, 'r--', label='Emisión original')
        
        # Graficar la emisión medida normalizada
        plt.plot(wave_lengths, normalized_pixels[i], 'b-', label=f'Medida a {factor}x')
        
        # Calcular error cuadrático medio (RMSE) entre la emisión original y la medida
        error_mse = np.mean((spd_norm - normalized_pixels[i]) ** 2)
        error_rmse = np.sqrt(error_mse)
        error_porcentaje = error_rmse * 100  # Convertir a porcentaje
        error_espectral.append(error_porcentaje)
        
        plt.title(f'Comparación a distancia {factor}x (Error: {error_porcentaje:.2f}%)')
        plt.xlabel('Longitud de onda (nm)')
        plt.ylabel('Emisión normalizada')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return error_espectral
