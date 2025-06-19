#!/usr/bin/env python3
"""
Script para analizar la irradiancia total de un emisor de área
a diferentes distancias manteniendo un FOV fijo.
"""
import os
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

# Establecer variante spectral de Mitsuba
mi.set_variant('cuda_ad_spectral')

# Importar utilidades
from src.sensor_utils import load_fixed_sensor_gaussian
from src.emission_utils import load_spd_data
from src.scene_utils import create_area_emitter_scene
from src.visualization_utils import ensure_output_dir, visualize_images, analyze_spectral_comparison, detailed_comparison, analyze_intensity_falloff

# Configuración de parámetros
wave_lengths = np.linspace(380, 780, 49).astype(int)  # Espectro visible
sigma = 10
k = 3
n = 15
fov = 40  # FOV fijo en grados
base_distance = 15  # distancia base (1x)
distance_factors = [1, 2, 4, 8, 16]  # Factores de distancia
sphere_radius = 5.0  # Radio de la esfera emisora
desarrollo = True  # Modo desarrollo
spp = 1024 if desarrollo else 4096  # muestras por píxel

# Crear directorio de salida si no existe
ensure_output_dir()

# Cargar datos espectrales del emisor
data_file = 'mitsuba/test.spd'
spd_wavelengths, spd_values = load_spd_data(data_file)

# Crear la escena con una esfera emisora
scene = create_area_emitter_scene(spd_wavelengths, spd_values, sphere_radius)

# Lista para almacenar irradiancias totales
irradiancias = []
imagenes = []
pixeles = []

# Renderizar y calcular irradiancia total para cada distancia
print("Calculando irradiancia total para diferentes distancias (FOV fijo)...")
for factor in distance_factors:
    r = base_distance * factor
    print(f"- Distancia {factor}x (r = {r}): creando sensor...")
    # Sensor fijo en el eje Z negativo mirando al origen
    sensor = load_fixed_sensor_gaussian(
        0.0, 0.0, -r,
        0.0, 0.0, 0.0,
        wave_lengths=wave_lengths,
        sigma=sigma,
        fov=fov,
        k=k,
        n_wavelents=n
    )
    img = mi.render(scene, sensor=sensor, spp=spp)
    # Sumar valores de la imagen para obtener la irradiancia total
    irr_total = img.numpy().sum()
    irradiancias.append(irr_total)
    # Guardar imagen y píxel central para análisis visual
    imagenes.append(img)
    pixel = img[128, 128, :].numpy()
    pixeles.append(pixel)
    print(f"  Irradiancia total = {irr_total:.6e}\n")

# Graficar irradiancia total vs distancia
plt.figure(figsize=(10, 6))
plt.plot(distance_factors, irradiancias, 'o-', linewidth=2, markersize=8)
plt.xscale('log')
plt.yscale('log')
plt.title('Irradiancia total vs distancia (FOV fijo)')
plt.xlabel('Factor de distancia')
plt.ylabel('Irradiancia total')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('output/fixed_fov_total_irradiance.png', dpi=150)
plt.show()

# Comparación con ley 1/r^2
irr_rel = [irr / irradiancias[0] for irr in irradiancias]
ley_inv = [1 / (f**2) for f in distance_factors]

plt.figure(figsize=(10, 6))
plt.plot(distance_factors, irr_rel, 'o-', label='Irradiancia relativa', linewidth=2)
plt.plot(distance_factors, ley_inv, 's--', label='Ley 1/r²', linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.title('Comparación irradiancia relativa vs ley 1/r²')
plt.xlabel('Factor de distancia')
plt.ylabel('Irradiancia relativa')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('output/fixed_fov_irradiance_falloff.png', dpi=150)
plt.show()

# === Gráficas de visualización adicionales con FOV fijo ===
# Mostrar imágenes a diferentes distancias
visualize_images(
    imagenes,
    distance_factors,
    banda=1,
    title_prefix='Emisor Área FOV fijo -',
    save_path='output/fixed_fov_images.png'
)

# Análisis espectral comparativo
normalized_pixels = analyze_spectral_comparison(
    pixeles,
    wave_lengths,
    spd_wavelengths,
    spd_values,
    distance_factors,
    title_suffix='- FOV fijo',
    save_paths={
        'spectra': 'output/fixed_fov_spectra.png',
        'comparison': 'output/fixed_fov_spectra_comparison.png'
    }
)

# Comparación detallada entre emisión original y medida
error_espectral = detailed_comparison(
    wave_lengths,
    normalized_pixels,
    spd_wavelengths,
    spd_values,
    distance_factors,
    save_path='output/fixed_fov_detail.png'
)

# Análisis de caída de intensidad
intensidades2, intensidad_rel2, error_porcentual2 = analyze_intensity_falloff(
    pixeles,
    distance_factors,
    title_suffix='- FOV fijo',
    save_paths={
        'intensity_falloff': 'output/fixed_fov_intensity_falloff.png',
        'error': 'output/fixed_fov_inverse_square_error.png'
    }
)
