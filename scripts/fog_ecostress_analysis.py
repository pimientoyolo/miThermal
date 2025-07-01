#!/usr/bin/env python3
"""
Script para analizar el efecto de la niebla usando datos espectrales de ECOSTRESS.
Utiliza los datos de absorción de agua destilada como coeficiente de extinción
para simular niebla realista a diferentes distancias.

Basado en fog_distance_analysis.py pero usando datos reales de ECOSTRESS.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

import mitsuba as mi
from scipy import constants as const
from src.visualization_utils import ensure_output_dir
from src.ecostress_utils import load_ecostress_material, reflectance_to_absorptance

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
print("Configurando análisis de niebla con datos ECOSTRESS...")
print("Usando datos de absorción de agua destilada para simular niebla realista")

# Longitudes de onda en LWIR (8000-14000 nm)
wave_lengths = np.linspace(8000, 14000, 49).astype(int)
sigma = 10
phi = -45  # Ángulo azimutal fijo
theta = 0  # Ángulo polar fijo
k = 3
n = 15

# Distancias de la cámara (factores de distancia base)
base_distance = 12  # Distancia base
distance_factors = [1, 2, 4, 8]  # Factores: 1x, 2x, 4x, 8x
distances = [base_distance * factor for factor in distance_factors]

# Configuración de muestreo
desarrollo = True
spp = 1024 if desarrollo else 4096

print(f"Distancia base: {base_distance}")
print(f"Factores de distancia: {distance_factors}")
print(f"Distancias reales: {distances}")

# %%
# Cargar datos de ECOSTRESS para agua destilada
print("Cargando datos de ECOSTRESS para agua destilada...")

# water_file = "ecostress/water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt"
water_file = "ecostress/manmade.generalconstructionmaterial.paint.solid.all.0385uuupnt.jhu.becknic.spectrum.txt"  # Usar un archivo de ejemplo

try:
    # Cargar datos de reflectancia del agua usando ECOSTRESS
    wavelengths_nm, reflectance, emissivity, sigma_t, metadata = load_ecostress_material(water_file)
    
    # Convertir nm a μm solo para mostrar en prints (ECOSTRESS está originalmente en μm)
    wavelengths_um_display = wavelengths_nm / 1000.0
    
    # Interpolar sigma_t a nuestras longitudes de onda LWIR (ambos arrays ya están en nm)
    sigma_t_interp = np.interp(wave_lengths, wavelengths_nm, sigma_t)
    
    # Escalar los coeficientes sigma_t para obtener valores realistas para niebla
    # Los valores de sigma_t del agua pura son muy grandes, así que los escalamos hacia abajo
    scale_factor = 1e-3  # Factor de escala para simular niebla de agua (más realista físicamente)
    sigma_t_list = sigma_t_interp * scale_factor
    
    print(f"Material: {metadata.get('Name', 'Agua destilada')}")
    print(f"Rango de longitudes de onda ECOSTRESS: {wavelengths_um_display[0]:.3f} - {wavelengths_um_display[-1]:.3f} μm")
    print(f"Puntos de datos ECOSTRESS: {len(wavelengths_nm)}")
    print(f"σₜ medio original: {np.mean(sigma_t):.2e} m⁻¹")
    print(f"σₜ medio escalado para niebla: {np.mean(sigma_t_list):.4f} m⁻¹")
    print(f"Factor de escala aplicado: {scale_factor}")
    
    # Visualizar los datos de sigma_t
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.semilogy(wavelengths_um_display, sigma_t, 'b-', linewidth=2, label='σₜ original')
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel('σₜ (m⁻¹, escala log)')
    plt.title(f'Coeficiente de extinción - {metadata.get("Name", "Agua")}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogy(wave_lengths / 1000.0, sigma_t_list, 'r-', linewidth=2, label=f'σₜ escalado (×{scale_factor})')
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel('σₜ (m⁻¹, escala log)')
    plt.title('Coeficiente de extinción para niebla')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
except Exception as e:
    print(f"Error cargando datos de ECOSTRESS: {e}")
    print("Usando coeficiente de extinción constante como respaldo")
    sigma_t_value = 0.01
    sigma_t_list = [sigma_t_value] * len(wave_lengths)
    sigma_t_list = [sigma_t_value] * len(wave_lengths)

# %%
# Cargar datos de emisividad para el material emisor
print("Cargando base de datos de materiales para el emisor...")

try:
    dataBaseName = np.load("data/matName_FullDatabase.npy", allow_pickle=True).item()["matName"]
    dataBaseName = dataBaseName.squeeze()
    dataBaseName = np.hstack(dataBaseName)
    
    dataBaseLib = np.load("data/matLib_FullDatabase.npy", allow_pickle=True).item()["matLib"]
    dataBaseLib = dataBaseLib[::-1, :]
    
    # Usar material stone para el emisor
    material = "stone"
    indice_material = np.where(dataBaseName == material)[0][0]
    firma = dataBaseLib[:, indice_material]
    
    print(f"Material del emisor: {material}")
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

print(f"Temperatura del objeto emisor: {temperatura} K")

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
# Definir medio de niebla usando datos de absorción del agua
print("Configurando medio de niebla con datos espectrales de agua...")

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
    "phase": {
        "type": "hg",
        "g": 0.95,  # Factor de anisotropía para LWIR
    },
}

print("Coeficientes de extinción (sigma_t):")
print(f"  Mínimo: {np.min(sigma_t_list):.6f}")
print(f"  Máximo: {np.max(sigma_t_list):.6f}")
print(f"  Promedio: {np.mean(sigma_t_list):.6f}")

# %%
# Mostrar datos espectrales de absorción del agua
# Calcular coeficiente de absorción y su interpolación
absorption_coefficient = reflectance_to_absorptance(reflectance)
absorption_interp = np.interp(wave_lengths, wavelengths_nm, absorption_coefficient)
plt.figure(figsize=(15, 10))

# Datos originales de ECOSTRESS
if 'wavelengths_nm' in locals():
    plt.subplot(2, 2, 1)
    plt.plot(wavelengths_um_display, reflectance, 'b-', linewidth=2)
    plt.title('Reflectancia del agua destilada (ECOSTRESS)')
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel('Reflectancia (fracción)')
    plt.grid(True, alpha=0.3)
    plt.xlim(8, 14)
    
    plt.subplot(2, 2, 2)
    plt.plot(wavelengths_um_display, absorption_coefficient, 'r-', linewidth=2)
    plt.title('Coeficiente de absorción del agua')
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel('Absorción (fracción)')
    plt.grid(True, alpha=0.3)
    plt.xlim(8, 14)
    
    plt.subplot(2, 2, 3)
    wave_lengths_um_display = wave_lengths / 1000.0  # Convertir nm a μm para display
    plt.plot(wave_lengths_um_display, absorption_interp, 'g-', linewidth=2, label='Interpolado')
    plt.title('Absorción interpolada a LWIR')
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel('Absorción (fracción)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(wave_lengths, sigma_t_list, 'm-', linewidth=2)
    plt.title('Sigma_t para simulación de niebla')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Sigma_t')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/ecostress_water_absorption.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Integradores
integrador_niebla = {"type": "volpathmis", "max_depth": 8}
integrador_sin_niebla = {"type": "path"}

# %%
# Renderizar escenas a diferentes distancias
print("\nRenderizando escenas con niebla espectral de agua...")

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
    plt.title(f'Con niebla H₂O - {factor}x\n(Banda {wave_lengths[banda]} nm)')
    plt.axis('off')
    plt.colorbar()
    
    # Sin niebla
    plt.subplot(2, len(distance_factors), i + len(distance_factors) + 1)
    plt.imshow(imagenes_sin_niebla[i][:, :, banda], cmap='gray', vmin=0, vmax=max_val)
    plt.title(f'Sin niebla - {factor}x\n(Banda {wave_lengths[banda]} nm)')
    plt.axis('off')
    plt.colorbar()

plt.tight_layout()
plt.savefig('output/fog_ecostress_images.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualización 2: Firmas espectrales para cada distancia
plt.figure(figsize=(20, 12))

# Espectros con niebla
plt.subplot(2, 2, 1)
for i, factor in enumerate(distance_factors):
    plt.plot(wave_lengths, pixeles_niebla[i], label=f'{factor}x', linewidth=2)
plt.title('Firmas espectrales con niebla de agua (ECOSTRESS)')
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
plt.title('Transmitancia medida vs distancia (Agua)')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Transmitancias teóricas
plt.subplot(2, 2, 4)
for i, factor in enumerate(distance_factors):
    plt.plot(wave_lengths, transmitancias_teoricas[i], label=f'{factor}x', linewidth=2)
plt.title('Transmitancia teórica vs distancia (Agua)')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('output/fog_ecostress_spectra.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualización 3: Comparación transmitancia medida vs teórica
plt.figure(figsize=(15, 10))

for i, factor in enumerate(distance_factors):
    plt.subplot(2, 2, i + 1)
    plt.plot(wave_lengths, transmitancias_medidas[i], 'b-', label='Medida', linewidth=2)
    plt.plot(wave_lengths, transmitancias_teoricas[i], 'r--', label='Teórica (H₂O)', linewidth=2)
    plt.title(f'Transmitancia - Distancia {factor}x\n(Niebla de agua)')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Transmitancia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('output/fog_ecostress_transmittance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualización 4: Análisis de atenuación espectral específica del agua
plt.figure(figsize=(20, 10))

# Análisis espectral de la atenuación
plt.subplot(2, 3, 1)
plt.plot(wave_lengths, sigma_t_list, 'b-', linewidth=2)
plt.title('Coeficiente de extinción espectral (H₂O)')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Sigma_t')
plt.grid(True, alpha=0.3)

# Atenuación por banda espectral
plt.subplot(2, 3, 2)
intensidades_totales_niebla = [np.sum(pixel) for pixel in pixeles_niebla]
intensidades_totales_sin_niebla = [np.sum(pixel) for pixel in pixeles_sin_niebla]
transmitancias_totales = np.array(intensidades_totales_niebla) / np.array(intensidades_totales_sin_niebla)

plt.plot(distance_factors, transmitancias_totales, 'go-', label='Medida (Total)', linewidth=2, markersize=8)
plt.title('Transmitancia total vs Distancia')
plt.xlabel('Factor de distancia')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)

# Comparación con Beer-Lambert promedio
plt.subplot(2, 3, 3)
sigma_t_promedio = np.mean(sigma_t_list)
distancias_teoricas = np.array(distancias_reales)
transmitancias_beer_lambert = np.exp(-sigma_t_promedio * distancias_teoricas)

plt.plot(distance_factors, transmitancias_totales, 'go-', label='Medida', linewidth=2, markersize=8)
plt.plot(distance_factors, transmitancias_beer_lambert, 'r--', label='Beer-Lambert (σₜ promedio)', linewidth=2)
plt.title('Comparación con Beer-Lambert')
plt.xlabel('Factor de distancia')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)

# Análisis por longitud de onda específica
bandas_interes = [0, 12, 24, 36, 48]  # Seleccionar algunas bandas
plt.subplot(2, 3, 4)
for banda in bandas_interes:
    transmitancias_banda = [trans[banda] for trans in transmitancias_medidas]
    plt.plot(distance_factors, transmitancias_banda, 'o-', 
             label=f'{wave_lengths[banda]} nm', linewidth=2, markersize=6)
plt.title('Transmitancia por banda espectral')
plt.xlabel('Factor de distancia')
plt.ylabel('Transmitancia')
plt.legend()
plt.grid(True, alpha=0.3)

# Diferencia espectral entre distancias
plt.subplot(2, 3, 5)
for i, factor in enumerate(distance_factors[1:], 1):
    diff = np.array(pixeles_sin_niebla[0]) - np.array(pixeles_niebla[i])
    plt.plot(wave_lengths, diff, label=f'Sin niebla - {factor}x', linewidth=2)
plt.title('Diferencia espectral por atenuación')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Diferencia de intensidad')
plt.legend()
plt.grid(True, alpha=0.3)

# Error espectral
plt.subplot(2, 3, 6)
for i, factor in enumerate(distance_factors):
    error = np.abs(transmitancias_medidas[i] - transmitancias_teoricas[i]) / transmitancias_teoricas[i] * 100
    plt.plot(wave_lengths, error, label=f'{factor}x', linewidth=2)
plt.title('Error relativo vs teórico (%)')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Error (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/fog_ecostress_spectral_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Tabla de resultados
print("\nResultados del análisis de niebla con datos ECOSTRESS de agua")
print("=" * 110)
print(f"{'Distancia':^12} | {'Dist. real':^12} | {'Int. niebla':^15} | {'Int. sin niebla':^15} | {'Trans. total':^15} | {'Trans. teórica':^15} | {'Error %':^10}")
print("-" * 110)

# Calcular transmitancia teórica con sigma_t promedio
error_promedio = []
for i, factor in enumerate(distance_factors):
    dist_real = distancias_reales[i]
    int_niebla = intensidades_totales_niebla[i]
    int_sin_niebla = intensidades_totales_sin_niebla[i]
    trans_total = transmitancias_totales[i]
    trans_teorica = transmitancias_beer_lambert[i]
    error_pct = np.abs(trans_total - trans_teorica) / trans_teorica * 100
    error_promedio.append(error_pct)
    
    print(f"{factor:^12.0f}x | {dist_real:^12.1f} | {int_niebla:^15.6f} | {int_sin_niebla:^15.6f} | {trans_total:^15.6f} | {trans_teorica:^15.6f} | {error_pct:^10.2f}%")

print("=" * 110)
print(f"Sigma_t promedio del agua: {sigma_t_promedio:.6f}")

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
    'error_promedio': error_promedio,
    'sigma_t_list': sigma_t_list.tolist() if isinstance(sigma_t_list, np.ndarray) else sigma_t_list,
    'sigma_t_promedio': sigma_t_promedio,
    'temperatura': temperatura,
    'material': material if 'material' in locals() else 'unknown',
    'water_file': water_file,
    'scale_factor': scale_factor if 'scale_factor' in locals() else None
}

with open('output/fog_ecostress_results.pkl', 'wb') as f:
    pickle.dump(resultados, f)

print("Resultados guardados en 'output/fog_ecostress_results.pkl'")
print("\nAnálisis con datos ECOSTRESS completado!")
print("Archivos generados:")
print("- output/ecostress_water_absorption.png")
print("- output/fog_ecostress_images.png")
print("- output/fog_ecostress_spectra.png") 
print("- output/fog_ecostress_transmittance_comparison.png")
print("- output/fog_ecostress_spectral_analysis.png")
print("- output/fog_ecostress_results.pkl")
