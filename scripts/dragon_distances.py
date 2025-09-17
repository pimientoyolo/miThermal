# %%
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

# Establecer variante spectral de Mitsuba
mi.set_variant('cuda_ad_spectral')

# Importar módulos de utilidad
from src.sensor_utils import load_sensor_gaussian
from src.emission_utils import blackbody_radiance_nm, load_material_data
from src.scene_utils import create_dragon_scene
from src.visualization_utils import (analyze_spectral_comparison, analyze_intensity_falloff, 
                                   visualize_images, detailed_comparison)

# %%
# Configuración inicial
wave_lengths = np.linspace(8000, 14000, 49).astype(int)  # Longitudes de onda para el dragón (infrarrojo)
sigma = 10
phi = 0
theta = 0
k = 3
n = 15
temperatura = 5000  # Temperatura del objeto (K)

# %%
# Cargar datos de emisividad
dataBaseName, dataBaseLib = load_material_data()

# Configurar material
material = "stone"  # Material seleccionado
indice_material = np.where(dataBaseName == material)[0][0]
firma = dataBaseLib[:, indice_material]  # Firma espectral del material

# Calcular la emisión
black_body = blackbody_radiance_nm(wave_lengths, temperatura)  # Radiancia del objeto
emision = black_body * firma  # Emisión del objeto

# %%
# Crear escena con el dragón
scene = create_dragon_scene(wave_lengths, emision, material)

# %%
# Definir los factores de distancia para el dragón
distancias = [1, 2, 3, 5, 10]  # Multiplicadores de distancia: 1x, 2x, 3x, 5x, 10x
base_distance = 10  # Distancia base (1x)

# Configuración de muestreo
desarrollo = True  # Cambiar a False para renderizado final dqe alta calidad
spp = 1024 if desarrollo else 4096  # Muestras por píxel según desarrollo o final

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
    sensor = load_sensor_gaussian(r, phi, theta, wave_lengths, sigma, 
                                 base_distance=base_distance, base_fov=40, k=k, n_wavelents=n)
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
    banda=1, 
    title_prefix='Dragón -',
    save_path='output/dragon_distances_images.png'
)

# %%
# Crear rutas para guardar gráficas
save_paths = {
    'spectra': 'output/dragon_distances_spectra.png',
    'comparison': 'output/dragon_distances_comparison.png'
}

# Análisis de datos espectrales
normalized_pixels = analyze_spectral_comparison(
    pixeles, 
    wave_lengths, 
    wave_lengths, 
    firma * np.max(black_body),  # Para mostrar la firma original
    distancias,
    title_suffix='- Emisor Dragón', 
    save_paths=save_paths
)

# %%
# Análisis de la caída de intensidad
intensidades, intensidad_relativa, error_porcentual = analyze_intensity_falloff(
    pixeles, 
    distancias, 
    title_suffix='- Emisor Dragón', 
    save_paths={
        'intensity_falloff': 'output/dragon_intensity_falloff.png',
        'error': 'output/dragon_inverse_square_error.png'
    }
)

# %%
# Comparación detallada entre emisión original y medida
error_espectral = detailed_comparison(
    wave_lengths,
    normalized_pixels,
    wave_lengths,
    firma * np.max(black_body),
    distancias,
    'output/dragon_comparison_detail.png'
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
    'firma': firma.tolist(),
    'temperatura': temperatura
}

# Guardar resultados
with open('output/dragon_results.pkl', 'wb') as f:
    pickle.dump(resultados, f)

print("Resultados guardados en 'output/dragon_results.pkl'")
