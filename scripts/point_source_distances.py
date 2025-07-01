# %%
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

# Establecer variante spectral de Mitsuba
mi.set_variant('cuda_ad_spectral')

# Importar módulos de utilidad
from src.sensor_utils import load_sensor_gaussian
from src.emission_utils import load_spd_data
from src.scene_utils import create_point_source_scene
from src.visualization_utils import (analyze_spectral_comparison, analyze_intensity_falloff, 
                                   visualize_images, detailed_comparison)

# %%
# Configuración inicial
# Usamos longitudes de onda en el mismo rango que test.spd (380-780 nm)
wave_lengths = np.linspace(380, 780, 49).astype(int)
sigma = 10
phi = 0
theta = 0
k = 3
n = 15
# %%
# Cargar los datos de emisión espectral desde el archivo test.spd
spd_wavelengths, spd_values = load_spd_data('mitsuba/test.spd')

# %%
# Crear la escena con un emisor puntual y una esfera de referencia
scene = create_point_source_scene(spd_wavelengths, spd_values)

# %%
# Definir los factores de distancia (diferentes a los del dragón)
distancias = [1, 8, 16, 32, 64]   # multiplicadores de distancia: 1x, 2x, 4x, 8x, 16x
base_distance = 35  # distancia base (1x)

# Configuración de muestreo
desarrollo = True  # Cambiar a False para renderizado final de alta calidad
spp = 1024 if desarrollo else 4096  # Menos muestras durante desarrollo, más para resultado final

# Lista para almacenar los resultados
imagenes = []
sensores = []
pixeles = []

# Renderizar la escena para cada distancia
for factor in distancias:
    # Calcular la nueva distancia
    r = base_distance * factor
    
    # Crear un sensor a esta distancia, ajustando el FOV para mantener el ángulo sólido constante
    print(f"Renderizando imagen a distancia {factor}x ({r} unidades)...")
    # Pasamos la distancia base para el cálculo del FOV ajustado
    sensor = load_sensor_gaussian(r, phi, theta, wave_lengths, sigma, base_distance=base_distance, base_fov=40, k=k, n_wavelents=n)
    sensores.append(sensor)
    
    # Renderizar la escena con este sensor
    imagen = mi.render(scene, sensor=sensor, spp=spp)
    imagenes.append(imagen)
    
    # Extraer valores del pixel central
    x, y = 128, 128  # centro de la imagen
    pixel = imagen[y, x, :].numpy()
    pixeles.append(pixel)

# %%
# Visualizar las imágenes a diferentes distancias
visualize_images(
    imagenes, 
    distancias, 
    banda=0, 
    title_prefix='Emisor Puntual -',
    save_path='output/point_emitter_distances_images.png'
)

# %%
# Crear rutas para guardar gráficas
save_paths = {
    'spectra': 'output/point_emitter_distances_spectra.png',
    'comparison': 'output/point_emitter_distances_comparison.png'
}

# Análisis de datos espectrales
normalized_pixels = analyze_spectral_comparison(
    pixeles, 
    wave_lengths, 
    spd_wavelengths, 
    spd_values,
    distancias,
    title_suffix='- Emisor Puntual', 
    save_paths=save_paths
)

# %%
# Comparación detallada entre emisión original y medida
error_espectral = detailed_comparison(
    wave_lengths,
    normalized_pixels,
    spd_wavelengths,
    spd_values,
    distancias,
    'output/point_emitter_comparison_detail.png'
)

# %%
# Análisis de la caída de intensidad
intensidades, intensidad_relativa, error_porcentual = analyze_intensity_falloff(
    pixeles, 
    distancias, 
    title_suffix='- Emisor Puntual', 
    save_paths={
        'intensity_falloff': 'output/point_emitter_intensity_falloff.png',
        'error': 'output/point_emitter_inverse_square_error.png'
    }
)

# %%
# Guardar resultados para comparación posterior
import pickle
import os

# Asegurar que el directorio existe
if not os.path.exists('output'):
    os.makedirs('output')

# Crear diccionario con todos los resultados relevantes
resultados = {
    'distancias': distancias,
    'intensidades': intensidades,
    'intensidad_relativa': intensidad_relativa,
    'error_porcentual': error_porcentual,
    'error_espectral': error_espectral,
    'pixeles_normalizados': normalized_pixels,
    'wave_lengths': wave_lengths.tolist(),
    'spd_wavelengths': spd_wavelengths.tolist(),
    'spd_values': spd_values.tolist()
}

# Guardar resultados
with open('output/point_emitter_results.pkl', 'wb') as f:
    pickle.dump(resultados, f)

print("Resultados guardados en 'output/point_emitter_results.pkl'")
