"""
Script para crear una escena con 4 esferas emisoras a diferentes distancias
con una cámara fija ligeramente a la derecha.

Las esferas están ubicadas a distancias 1x, 2x, 4x y 8x de la distancia base.
"""

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

# Establecer variante spectral de Mitsuba
mi.set_variant('cuda_ad_spectral')

# Importar módulos de utilidad
from src.sensor_utils import load_fixed_sensor_gaussian
from src.emission_utils import load_spd_data
from src.scene_utils import create_multi_sphere_scene
from src.visualization_utils import ensure_output_dir

# %%
# Configuración inicial
wave_lengths = np.linspace(380, 780, 41).astype(int)  # Espectro visible
sigma = 10
k = 3
n = 15

# Distancias de las esferas
base_distance = 3  # Distancia base
distance_factors = [1, 8, 32, 64]  # Factores de distancia
sphere_radius = 7.0  # Radio de las esferas

# Configuración de muestreo
desarrollo = True  # Cambiar a False para renderizado final de alta calidad
spp = 512 if desarrollo else 2048  # Muestras por píxel

# %%
# Cargar datos espectrales del archivo test.spd
spd_wavelengths, spd_values = load_spd_data('mitsuba/test.spd')

# %%
# Crear la escena con múltiples esferas
print("Creando escena con 4 esferas a diferentes distancias...")
scene = create_multi_sphere_scene(
    spd_wavelengths, 
    spd_values, 
    base_distance=base_distance,
    sphere_radius=sphere_radius,
    distance_factors=distance_factors
)

# %%
# Crear sensor con cámara fija ligeramente a la derecha
print("Configurando cámara fija ligeramente a la derecha...")

# Posición de la cámara: ligeramente a la derecha y alejada para ver todas las esferas
camera_x = 50  # Desplazamiento hacia la derecha
camera_y = 5   # Ligeramente elevada
camera_z = -40 * base_distance * 0.8  # Alejada proporcionalmente para ver todas las esferas

# Punto hacia donde mira (centro de las esferas, aproximadamente)
target_x = 0
target_y = 0
target_z = base_distance * (max(distance_factors) + min(distance_factors)) / 2  # Punto medio entre las esferas

sensor = load_fixed_sensor_gaussian(
    camera_x, camera_y, camera_z,
    target_x, target_y, target_z,
    wave_lengths=wave_lengths, 
    sigma=sigma,
    fov=60,  # FOV amplio para capturar todas las esferas
    k=k, n_wavelents=n
)

# %%
# Renderizar la escena
print("Renderizando escena con múltiples esferas...")
imagen = mi.render(scene, sensor=sensor, spp=spp)

# %%
# Asegurar que existe el directorio de salida
ensure_output_dir()

# Visualizar la imagen renderizada
plt.figure(figsize=(12, 10))

# Mostrar diferentes bandas espectrales
bandas_interes = [0, 10, 20, 30]  # Diferentes bandas para visualizar
nombres_bandas = [f'{wave_lengths[b]} nm' for b in bandas_interes]

for i, banda in enumerate(bandas_interes):
    plt.subplot(2, 2, i + 1)
    plt.imshow(imagen[:, :, banda], cmap='hot')
    plt.colorbar()
    plt.title(f'Banda espectral: {nombres_bandas[i]}')
    plt.axis('off')

plt.suptitle('Escena con 4 esferas emisoras a diferentes distancias\n(Cámara fija ligeramente a la derecha)', fontsize=14)
plt.tight_layout()
plt.savefig('output/multi_sphere_scene_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Análisis de intensidades en diferentes regiones de la imagen
print("\nAnalizando intensidades de las diferentes esferas...")

# Generar regiones dinámicamente basándose en los factores de distancia
# Las coordenadas Y se calculan aproximadamente basándose en la perspectiva
# (Las esferas más lejanas aparecen más arriba en la imagen)
regiones = {}
for i, factor in enumerate(distance_factors):
    # Calcular coordenada Y basándose en la distancia relativa
    # Esferas más cercanas aparecen más abajo (Y mayor)
    # Esferas más lejanas aparecen más arriba (Y menor)
    y_base = 200
    y_offset = i * -25  # Cada esfera se desplaza 25 píxeles hacia arriba
    y_coord = max(50, y_base + y_offset)  # No ir más arriba de Y=50
    
    regiones[f'{factor}x'] = (128, y_coord)  # X centrado, Y calculado
    print(f"Región para esfera {factor}x: pixel ({128}, {y_coord})")

intensidades_regiones = {}
for nombre, (x, y) in regiones.items():
    # Extraer pixel y calcular intensidad promedio
    pixel = imagen[y, x, :].numpy()
    intensidad = np.mean(pixel)
    intensidades_regiones[nombre] = intensidad
    print(f"Esfera {nombre}: Intensidad promedio = {intensidad:.6f}")

# %%
# Gráfica de intensidades vs distancia
distancias_reales = [base_distance * f for f in distance_factors]
intensidades = [intensidades_regiones[f'{f}x'] for f in distance_factors]

plt.figure(figsize=(10, 6))
plt.plot(distance_factors, intensidades, 'o-', label='Intensidad medida', linewidth=2, markersize=8)

# Ley del cuadrado inverso teórica (normalizada al primer punto)
ley_cuadrado_inverso = [intensidades[0] / (f**2) for f in distance_factors]
plt.plot(distance_factors, ley_cuadrado_inverso, 's--', label='Ley del cuadrado inverso (1/r²)', 
         linewidth=2, markersize=8, alpha=0.7)

plt.xlabel('Factor de distancia')
plt.ylabel('Intensidad medida')
plt.title('Comparación de intensidades: 4 esferas a diferentes distancias')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xscale('log')
plt.savefig('output/multi_sphere_intensity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Análisis del error respecto a la ley del cuadrado inverso
intensidad_relativa = [i / intensidades[0] for i in intensidades]
ley_teorica = [1 / (f**2) for f in distance_factors]
error_porcentual = [(medido / teorico - 1) * 100 for medido, teorico in zip(intensidad_relativa, ley_teorica)]

plt.figure(figsize=(10, 6))
plt.bar(range(len(distance_factors)), error_porcentual, 
        tick_label=[f'{f}x' for f in distance_factors])
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Error porcentual respecto a la ley del cuadrado inverso')
plt.xlabel('Factor de distancia')
plt.ylabel('Error (%)')
plt.grid(True, axis='y', alpha=0.3)
plt.savefig('output/multi_sphere_error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Tabla de resultados
print("\nResultados del análisis de múltiples esferas")
print("=" * 80)
print(f"{'Distancia':^12} | {'Dist. real':^12} | {'Intensidad':^15} | {'Intens. rel.':^15} | {'Ley 1/r²':^15} | {'Error %':^10}")
print("-" * 80)
for i, factor in enumerate(distance_factors):
    dist_real = distancias_reales[i]
    print(f"{factor:^12.0f}x | {dist_real:^12.1f} | {intensidades[i]:^15.8f} | {intensidad_relativa[i]:^15.8f} | {ley_teorica[i]:^15.8f} | {error_porcentual[i]:^10.2f}%")
print("=" * 80)

print("\nImágenes guardadas en:")
print("  - output/multi_sphere_scene_overview.png")
print("  - output/multi_sphere_intensity_analysis.png") 
print("  - output/multi_sphere_error_analysis.png")
