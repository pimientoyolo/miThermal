#!/usr/bin/env python3
"""
Script para analizar el efecto de la niebla (medio absorbente) a diferentes distancias.
Basado en Untitled-1 y la estructura de los scripts de emisores.

La cámara se coloca a distancias 1x, 2x, 4x, 8x para observar cómo la niebla
afecta la transmisión espectral a diferentes rangos.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

import mitsuba as mi
from scipy import constants as const
from src.visualization_utils import ensure_output_dir

# Establecer variante spectral de Mitsuba
mi.set_variant('cuda_ad_spectral')

from mitsuba import ScalarTransform4f as T


# %%
def load_sensor_spectral(r, phi, theta, wave_lengths, sigma, k=3, n_wavelents=5):
    """
    Load a sensor with spectral response.
    """
    z = r * np.sin(math.radians(theta))
    y = r * np.cos(math.radians(theta)) * np.sin(math.radians(phi))
    x = r * np.cos(math.radians(theta)) * np.cos(math.radians(phi))

    origin = np.array([x, y, z])

    film_dic = {
        "type": "specfilm",
        "width": 256,
        "height": 256,
        "rfilter": {"type": "tent"},
    }

    # Create sensor uniform
    distancia = wave_lengths[1] - wave_lengths[0]
    separar = distancia // 2

    for wave_length in wave_lengths:
        w_min = int(wave_length - separar)
        w_max = int(wave_length + separar)
        values = "1.0, 1.0"

        film_dic[f"band_{wave_length}"] = {
            "type": "regular",
            "wavelength_min": w_min,
            "wavelength_max": w_max,
            "values": values,
        }

    return {
        "type": "perspective",
        "fov": 40,
        "to_world": T().look_at(origin=origin, target=[0, 0, 0], up=[0, 0, 1]),
        "sampler": {"type": "independent", "sample_count": 1},
        "film": film_dic,
    }

def load_sensor_depth(r, phi, theta):
    """
    Load a depth sensor.
    """
    z = r * np.sin(math.radians(theta))
    y = r * np.cos(math.radians(theta)) * np.sin(math.radians(phi))
    x = r * np.cos(math.radians(theta)) * np.cos(math.radians(phi))

    origin = np.array([x, y, z])

    return {
        "type": "perspective",
        "fov": 40,
        "to_world": T().look_at(origin=origin, target=[0, 0, 0], up=[0, 0, 1]),
        "sampler": {"type": "independent", "sample_count": 1},
        "film": {
            "type": "hdrfilm",
            "width": 256,
            "height": 256,
            "rfilter": {"type": "box"},
            "pixel_format": "luminance",
            "component_format": "float32",
        },
    }

def blackbody_radiance_nm(wavelengths_nm, temperature):
    """
    Compute spectral radiance B(λ, T) of a black body using scipy constants.
    """
    wavelengths_m = np.array(wavelengths_nm, dtype=float) * 1e-9
    
    B_m = (
        (2 * const.h * const.c**2)
        / (wavelengths_m**5)
        / (np.exp(const.h * const.c / (wavelengths_m * const.k * temperature)) - 1)
    )
    
    B_nm = B_m * 1e-9
    return B_nm

def lista_a_string(valores):
    """
    Convert list to string format for Mitsuba.
    """
    return ", ".join(str(v) for v in valores)

# %%
# Configuración inicial
print("Configurando análisis de niebla a diferentes distancias...")
print("Usando LWIR (8-14 μm) para análisis de transmisión atmosférica")

# Longitudes de onda en LWIR (8000-14000 nm)
wave_lengths = np.linspace(8000, 14000, 49).astype(int)
sigma = 10
phi = -45  # Ángulo azimutal fijo
theta = 0  # Ángulo polar fijo
k = 3
n = 15

# Distancias de la cámara (factores de distancia base)
base_distance = 10  # Distancia base
distance_factors = [1, 2, 4, 8]  # Factores: 1x, 2x, 4x, 8x
distances = [base_distance * factor for factor in distance_factors]

# Configuración de muestreo
desarrollo = True
spp = 1024 if desarrollo else 4096

print(f"Distancia base: {base_distance}")
print(f"Factores de distancia: {distance_factors}")
print(f"Distancias reales: {distances}")

# %%
# Cargar datos de emisividad para el material
print("Cargando base de datos de materiales...")

try:
    dataBaseName = np.load("data/matName_FullDatabase.npy", allow_pickle=True).item()["matName"]
    dataBaseName = dataBaseName.squeeze()
    dataBaseName = np.hstack(dataBaseName)
    
    dataBaseLib = np.load("data/matLib_FullDatabase.npy", allow_pickle=True).item()["matLib"]
    dataBaseLib = dataBaseLib[::-1, :]
    
    # Usar material stone
    material = "stone"
    indice_material = np.where(dataBaseName == material)[0][0]
    firma = dataBaseLib[:, indice_material]
    
    print(f"Material seleccionado: {material}")
    print(f"Forma de la firma espectral: {firma.shape}")
    
except Exception as e:
    print(f"Error cargando base de datos: {e}")
    print("Usando emisividad constante de 0.8")
    firma = np.ones(len(wave_lengths)) * 0.8

# %%
# Crear emisión del objeto (cuerpo negro + emisividad)
temperatura = 5000  # Temperatura en Kelvin
black_body = blackbody_radiance_nm(wave_lengths, temperatura)
emision = black_body * firma

print(f"Temperatura del objeto: {temperatura} K")

# %%
# Crear esfera emisora
sphere = {
    "type": "sphere",
    "center": [0, 0, 0],
    "radius": 2.0,
    "bsdf": {
        "type": "diffuse",
        "reflectance": {
            "type": "rgb",
            "value": [0.8, 0.6, 0.2]  # Color dorado
        }
    },
    "emitter": {
        "type": "area",
        "radiance": {
            "type": "irregular",
            "wavelengths": lista_a_string(wave_lengths),
            "values": lista_a_string(emision),
        },
    },
}

# %%
# Definir medio de niebla
print("Configurando medio de niebla...")

# Configuración del coeficiente de extinción (sigma_t)
sigma_t_value = 0.01  # Coeficiente constante
sigma_t_list = [sigma_t_value] * len(wave_lengths)

print(str(wave_lengths[0]))
print(str(wave_lengths[-1]))


niebla = {
    "type": "homogeneous",
    "id": "niebla",
    "albedo": {
        "type": "constvolume",
        "value": {"type": "uniform", "value": 0.0},  # Solo absorción, sin scattering
    },
    "sigma_t": {
        "type": "constvolume",
        "value": {
            "type": "irregular",
            "wavelengths": lista_a_string(wave_lengths),
            "values": lista_a_string(sigma_t_list),
        },
    },
    "emitter": {
        "type": "area",
        "radiance": {
            "type": "regular",
            "wavelength_min": 8000,  # Longitud de onda mínima en nm
            "wavelength_max": 14000,
            "values": "0.2, 0.8", 
        },
    },
    "phase": {
        "type": "hg",
        "g": 0.95,  # Factor de anisotropía para LWIR
    },
}

print(f"Coeficiente de extinción sigma_t: {sigma_t_value}")

# %%
# Integradores
integrador_niebla = {"type": "volpathmis", "max_depth": 8}
integrador_sin_niebla = {"type": "path"}

# %%
# Renderizar escenas a diferentes distancias
print("\nRenderizando escenas...")

imagenes_niebla = []
imagenes_sin_niebla = []
imagenes_depth = []
distancias_reales = []
pixeles_niebla = []
pixeles_sin_niebla = []
transmitancias_medidas = []
transmitancias_teoricas = []

# Coordenadas del pixel a analizar (pixel central de la imagen 256x256)
x, y = 128, 128

for i, (factor, r) in enumerate(zip(distance_factors, distances)):
    print(f"\nProcesando distancia {factor}x ({r} unidades)...")
    
    # Crear sensores para esta distancia
    sensor_espectral_niebla = load_sensor_spectral(r, phi, theta, wave_lengths, sigma, k, n)
    sensor_espectral_niebla["medium"] = {"type": "ref", "id": "niebla"}
    
    sensor_espectral_sin_niebla = load_sensor_spectral(r, phi, theta, wave_lengths, sigma, k, n)
    sensor_depth = load_sensor_depth(r, phi, theta)
    
    # Crear escenas
    scene_niebla = mi.load_dict({
        "type": "scene",
        "integrator": integrador_niebla,
        "niebla": niebla,
        "sensor": sensor_espectral_niebla,
        "sphere": sphere,
    })
    
    scene_sin_niebla = mi.load_dict({
        "type": "scene",
        "integrator": integrador_sin_niebla,
        "sensor": sensor_espectral_sin_niebla,
        "sphere": sphere,
    })
    
    scene_depth = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "depth"},
        "sensor": sensor_depth,
        "sphere": sphere,
    })
    
    # Renderizar
    print("  Renderizando con niebla...")
    imagen_niebla = mi.render(scene_niebla, spp=spp)
    
    print("  Renderizando sin niebla...")
    imagen_sin_niebla = mi.render(scene_sin_niebla, spp=spp)
    
    print("  Renderizando mapa de profundidad...")
    imagen_depth = mi.render(scene_depth, spp=spp//4)
    
    # Almacenar imágenes
    imagenes_niebla.append(imagen_niebla)
    imagenes_sin_niebla.append(imagen_sin_niebla)
    imagenes_depth.append(imagen_depth)
    distancias_reales.append(r)
    
    # Extraer valores del pixel
    pixel_niebla = imagen_niebla[y, x, :].numpy()
    pixel_sin_niebla = imagen_sin_niebla[y, x, :].numpy()
    distancia_objeto_raw = imagen_depth[y, x].numpy()
    
    # Extraer valor escalar de la distancia
    if isinstance(distancia_objeto_raw, np.ndarray):
        distancia_objeto = float(distancia_objeto_raw.item())
    else:
        distancia_objeto = float(distancia_objeto_raw)
    
    pixeles_niebla.append(pixel_niebla)
    pixeles_sin_niebla.append(pixel_sin_niebla)
    
    # Calcular transmitancia
    transmitancia_medida = pixel_niebla / (pixel_sin_niebla + 1e-10)  # Evitar división por cero
    transmitancia_teorica = np.exp(-np.array(sigma_t_list) * distancia_objeto)
    
    transmitancias_medidas.append(transmitancia_medida)
    transmitancias_teoricas.append(transmitancia_teorica)
    
    print(f"  Distancia al objeto: {distancia_objeto:.2f}")
    print(f"  Intensidad media con niebla: {np.mean(pixel_niebla):.6f}")
    print(f"  Intensidad media sin niebla: {np.mean(pixel_sin_niebla):.6f}")
    print(f"  Transmitancia media: {np.mean(transmitancia_medida):.4f}")

# %%
# Crear directorio de salida
ensure_output_dir()

# %%
# Visualización 1: Imágenes con y sin niebla para cada distancia
print("\nGenerando visualizaciones...")

banda = 0  # Primera banda espectral para visualización
max_val_niebla = max([np.max(img[:, :, banda]) for img in imagenes_niebla])
max_val_sin_niebla = max([np.max(img[:, :, banda]) for img in imagenes_sin_niebla])
max_val = max(max_val_niebla, max_val_sin_niebla)

plt.figure(figsize=(20, 10))
for i, factor in enumerate(distance_factors):
    # Con niebla
    plt.subplot(2, len(distance_factors), i + 1)
    plt.imshow(imagenes_niebla[i][:, :, banda], cmap='gray', vmin=0, vmax=max_val)
    plt.title(f'Con niebla - {factor}x\n(Banda {wave_lengths[banda]} nm)')
    plt.axis('off')
    plt.colorbar()
    
    # Sin niebla
    plt.subplot(2, len(distance_factors), i + len(distance_factors) + 1)
    plt.imshow(imagenes_sin_niebla[i][:, :, banda], cmap='gray', vmin=0, vmax=max_val)
    plt.title(f'Sin niebla - {factor}x\n(Banda {wave_lengths[banda]} nm)')
    plt.axis('off')
    plt.colorbar()

plt.tight_layout()
plt.savefig('output/fog_distance_images.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualización 2: Firmas espectrales para cada distancia
plt.figure(figsize=(20, 12))

# Espectros con niebla
plt.subplot(2, 2, 1)
for i, factor in enumerate(distance_factors):
    plt.plot(wave_lengths, pixeles_niebla[i], label=f'{factor}x', linewidth=2)
plt.title('Firmas espectrales con niebla')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True, alpha=0.3)

# Espectros sin niebla
plt.subplot(2, 2, 2)
for i, factor in enumerate(distance_factors):
    plt.plot(wave_lengths, pixeles_sin_niebla[i], label=f'{factor}x', linewidth=2)
plt.title('Firmas espectrales sin niebla')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True, alpha=0.3)

# Transmitancias medidas
plt.subplot(2, 2, 3)
for i, factor in enumerate(distance_factors):
    plt.plot(wave_lengths, transmitancias_medidas[i], label=f'{factor}x', linewidth=2)
plt.title('Transmitancia medida vs distancia')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Transmitancias teóricas
plt.subplot(2, 2, 4)
for i, factor in enumerate(distance_factors):
    plt.plot(wave_lengths, transmitancias_teoricas[i], label=f'{factor}x', linewidth=2)
plt.title('Transmitancia teórica vs distancia')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('output/fog_distance_spectra.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualización 3: Comparación transmitancia medida vs teórica
plt.figure(figsize=(15, 10))

for i, factor in enumerate(distance_factors):
    plt.subplot(2, 2, i + 1)
    plt.plot(wave_lengths, transmitancias_medidas[i], 'b-', label='Medida', linewidth=2)
    plt.plot(wave_lengths, transmitancias_teoricas[i], 'r--', label='Teórica', linewidth=2)
    plt.title(f'Transmitancia - Distancia {factor}x')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Transmitancia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('output/fog_distance_transmittance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualización 4: Análisis de atenuación con la distancia
intensidades_totales_niebla = [np.sum(pixel) for pixel in pixeles_niebla]

intensidades_totales_sin_niebla = [np.sum(pixel) for pixel in pixeles_sin_niebla]
transmitancias_totales = np.array(intensidades_totales_niebla) / np.array(intensidades_totales_sin_niebla)

# Ley de Beer-Lambert: T = exp(-sigma_t * d)
distancias_teoricas = np.array(distancias_reales)
transmitancias_beer_lambert = np.exp(-sigma_t_value * distancias_teoricas)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(distance_factors, intensidades_totales_niebla, 'bo-', label='Con niebla', linewidth=2, markersize=8)
plt.plot(distance_factors, intensidades_totales_sin_niebla, 'ro-', label='Sin niebla', linewidth=2, markersize=8)
plt.title('Intensidad total vs Distancia')
plt.xlabel('Factor de distancia')
plt.ylabel('Intensidad total')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(distance_factors, transmitancias_totales, 'go-', label='Medida', linewidth=2, markersize=8)
plt.plot(distance_factors, transmitancias_beer_lambert, 'r--', label='Ley Beer-Lambert', linewidth=2)
plt.title('Transmitancia total vs Distancia')
plt.xlabel('Factor de distancia')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)

# Error respecto a Beer-Lambert
error_beer_lambert = np.abs(transmitancias_totales - transmitancias_beer_lambert) / transmitancias_beer_lambert * 100

plt.subplot(1, 3, 3)
plt.bar(range(len(distance_factors)), error_beer_lambert, 
        tick_label=[f'{f}x' for f in distance_factors])
plt.title('Error vs Ley de Beer-Lambert')
plt.xlabel('Factor de distancia')
plt.ylabel('Error (%)')
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/fog_distance_attenuation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Tabla de resultados
print("\nResultados del análisis de niebla a diferentes distancias")
print("=" * 100)
print(f"{'Distancia':^12} | {'Dist. real':^12} | {'Int. niebla':^15} | {'Int. sin niebla':^15} | {'Trans. total':^15} | {'Trans. Beer':^15} | {'Error %':^10}")
print("-" * 100)

for i, factor in enumerate(distance_factors):
    dist_real = distancias_reales[i]
    int_niebla = intensidades_totales_niebla[i]
    int_sin_niebla = intensidades_totales_sin_niebla[i]
    trans_total = transmitancias_totales[i]
    trans_beer = transmitancias_beer_lambert[i]
    error_pct = error_beer_lambert[i]
    
    print(f"{factor:^12.0f}x | {dist_real:^12.1f} | {int_niebla:^15.6f} | {int_sin_niebla:^15.6f} | {trans_total:^15.6f} | {trans_beer:^15.6f} | {error_pct:^10.2f}%")

print("=" * 100)

# %%
# Guardar resultados
print("\nGuardando resultados...")

resultados = {
    'distance_factors': distance_factors,
    'distances': distancias_reales,
    'wave_lengths': wave_lengths.tolist(),
    'pixeles_niebla': [p.tolist() for p in pixeles_niebla],
    'pixeles_sin_niebla': [p.tolist() for p in pixeles_sin_niebla],
    'transmitancias_medidas': [t.tolist() for t in transmitancias_medidas],
    'transmitancias_teoricas': [t.tolist() for t in transmitancias_teoricas],
    'intensidades_totales_niebla': intensidades_totales_niebla,
    'intensidades_totales_sin_niebla': intensidades_totales_sin_niebla,
    'transmitancias_totales': transmitancias_totales.tolist(),
    'transmitancias_beer_lambert': transmitancias_beer_lambert.tolist(),
    'error_beer_lambert': error_beer_lambert.tolist(),
    'sigma_t_value': sigma_t_value,
    'temperatura': temperatura,
    'material': material if 'material' in locals() else 'unknown'
}

with open('output/fog_distance_results.pkl', 'wb') as f:
    pickle.dump(resultados, f)

print("Resultados guardados en 'output/fog_distance_results.pkl'")
print("\nAnálisis completado!")
print("Archivos generados:")
print("- output/fog_distance_images.png")
print("- output/fog_distance_spectra.png") 
print("- output/fog_distance_transmittance_comparison.png")
print("- output/fog_distance_attenuation_analysis.png")
print("- output/fog_distance_results.pkl")
