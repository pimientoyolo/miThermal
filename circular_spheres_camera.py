#!/usr/bin/env python3
"""
Script para crear una escena con esferas emisoras cada 30 grados con emisión de cuerpo negro
y capturar imágenes LWIR (8-14 μm) con la cámara fija en el origen mirando hacia cada esfera.
Las temperaturas varían de -50°C a 250°C para mejor detección en el rango LWIR.
"""

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants as const

# Establecer variante spectral de Mitsuba
mi.set_variant('cuda_ad_spectral')

# Importar módulos de utilidad
from src.sensor_utils import load_fixed_sensor_gaussian
from src.visualization_utils import ensure_output_dir, lista_a_string


def blackbody_radiance_nm(wavelengths_nm, temperature):
    """
    Compute spectral radiance B(λ, T) of a black body using scipy constants.
    
    Args:
        wavelengths_nm: array-like of wavelengths in nanometers (nm).
        temperature:    temperature in Kelvin (K).
    
    Returns:
        numpy array of spectral radiance in W·sr⁻¹·m⁻²·nm⁻¹.
    """
    # Convert wavelengths to meters
    wavelengths_m = np.array(wavelengths_nm, dtype=float) * 1e-9
    
    # Planck's law for spectral radiance per meter: W·sr⁻¹·m⁻²·m⁻¹
    B_m = (2 * const.h * const.c**2) / (wavelengths_m**5) / (
        np.exp(const.h * const.c / (wavelengths_m * const.k * temperature)) - 1
    )
    
    # Convert from per meter to per nanometer: 1 m = 1e9 nm
    B_nm = B_m * 1e-9
    
    return B_nm

# %%
# Configuración inicial
# Rango LWIR: 8-14 micrómetros convertido a nanómetros
wave_lengths = np.linspace(8000, 14000, 41).astype(int)  # LWIR en nanómetros
sigma = 500  # Ajustado para LWIR
k = 3
n = 15
sphere_radius = 5.0  # Radio de las esferas
distance_from_center = 20.0  # Distancia desde el centro a las esferas
fov = 40  # Campo de visión

# Configuración de muestreo
desarrollo = False
spp = 1024 if desarrollo else 4096  # Muestras por píxel

# Ángulos cada 30 grados (0°, 30°, 60°, ..., 330°)
sphere_angles = np.arange(0, 360, 30)  # 12 esferas
camera_angles = np.arange(0, 360, 30)  # Capturar cada 30 grados (12 imágenes)

# Configuración de temperaturas: de -50°C a 250°C (rango más amplio para LWIR)
temp_min_celsius = -50
temp_max_celsius = 250
# Convertir a Kelvin
temp_min_kelvin = temp_min_celsius + 273.15  # 223.15 K
temp_max_kelvin = temp_max_celsius + 273.15  # 323.15 K

# Crear rango de temperaturas para cada esfera
temperatures_kelvin = np.linspace(temp_min_kelvin, temp_max_kelvin, len(sphere_angles))
temperatures_celsius = temperatures_kelvin - 273.15

print(f"Creando {len(sphere_angles)} esferas cada 30°")
print(f"Temperaturas de {temp_min_celsius}°C a {temp_max_celsius}°C (rango LWIR)")
print(f"Longitudes de onda: {wave_lengths[0]/1000:.1f} - {wave_lengths[-1]/1000:.1f} μm")
print(f"Capturando {len(camera_angles)} imágenes con cámara fija en origen")

# %%
# Cargar base de datos de firmas espectrales de materiales
print("\nCargando base de datos de materiales...")
try:
    dataBaseName = np.load('data/matName_FullDatabase.npy', allow_pickle=True).item()["matName"]
    dataBaseName = dataBaseName.squeeze() 
    dataBaseName = np.hstack(dataBaseName)  # lista de nombres de los materiales

    dataBaseLib = np.load('data/matLib_FullDatabase.npy', allow_pickle=True).item()["matLib"] 
    dataBaseLib = dataBaseLib[::-1, :]  # Reverso el orden de la base de datos
    
    print("Base de datos cargada exitosamente:")
    print(f"- {len(dataBaseName)} materiales disponibles")
    print(f"- Firmas espectrales: {dataBaseLib.shape}")
    
    # Usar concrete para todas las esferas para comparación directa de temperaturas
    if "concrete" in dataBaseName:
        material_base = "concrete"
        print(f"Material 'concrete' encontrado en la base de datos")
    else:
        # Buscar materiales similares a concrete
        materiales_concrete = [m for m in dataBaseName if "concrete" in m.lower()]
        if materiales_concrete:
            material_base = materiales_concrete[0]
            print(f"Material 'concrete' no encontrado, usando '{material_base}'")
        else:
            material_base = dataBaseName[0]  # Usar el primer material disponible
            print(f"No se encontró 'concrete', usando '{material_base}'")
    
    # Crear lista con el mismo material para todas las esferas
    materiales_disponibles = [material_base] * len(sphere_angles)
    
    print(f"Usando '{material_base}' para todas las {len(sphere_angles)} esferas")
    
except Exception as e:
    print(f"Error cargando base de datos de materiales: {e}")
    print("Usando solo emisores de cuerpo negro...")
    materiales_disponibles = [None] * len(sphere_angles)
    dataBaseName = None
    dataBaseLib = None

# %%
# Función para crear escena con múltiples esferas en círculo con emisores de cuerpo negro y materiales
def create_circular_spheres_scene(angles, temperatures, distance, radius):
    """
    Crea una escena con esferas emisoras distribuidas en un círculo con diferentes temperaturas
    y materiales (combinando cuerpo negro con emisividad de materiales).
    
    Args:
        angles: Ángulos en grados para posicionar las esferas
        temperatures: Temperaturas en Kelvin para cada esfera
        distance: Distancia desde el centro
        radius: Radio de las esferas
    """
    
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
        }
    }
    
    # Crear cada esfera en su posición angular con su temperatura específica
    for i, (angle_deg, temp_k) in enumerate(zip(angles, temperatures)):
        angle_rad = math.radians(angle_deg)
        x = distance * math.cos(angle_rad)
        z = distance * math.sin(angle_rad)  # Usar Z en lugar de Y para mantener en plano horizontal
        y = 0  # Todas las esferas a la misma altura
        
        # Calcular radiancia de cuerpo negro para esta temperatura
        blackbody_radiance = blackbody_radiance_nm(wave_lengths, temp_k)
        
        # Si tenemos materiales disponibles, combinar con emisividad del material
        if materiales_disponibles[i] is not None and dataBaseLib is not None:
            material_name = materiales_disponibles[i]
            # Buscar el índice del material en la base de datos
            indice_material = np.where(dataBaseName == material_name)[0][0]
            
            # Obtener la firma espectral del material (emisividad)
            # Nota: la base de datos puede tener diferentes longitudes de onda, interpolamos
            firma_material = dataBaseLib[:, indice_material]
            
            # Interpolar la firma del material a nuestras longitudes de onda LWIR
            # Asumimos que la base de datos está en un rango similar, sino habría que interpolar
            if len(firma_material) == len(wave_lengths):
                emisividad = firma_material
            else:
                # Interpolación simple - en un caso real habría que conocer las longitudes de onda de la base
                emisividad = np.interp(wave_lengths, 
                                     np.linspace(wave_lengths[0], wave_lengths[-1], len(firma_material)), 
                                     firma_material)
            
            # Combinar cuerpo negro con emisividad del material: L = ε * B(λ,T)
            emision_final = emisividad * blackbody_radiance
            sphere_name = f'sphere_{int(angle_deg)}deg_{int(temp_k-273.15)}C_{material_name}'
        else:
            # Solo cuerpo negro sin material específico
            emision_final = blackbody_radiance
            sphere_name = f'sphere_{int(angle_deg)}deg_{int(temp_k-273.15)}C_blackbody'
        
        scene_dict[sphere_name] = {
            'type': 'sphere',
            'center': [x, y, z],
            'radius': radius,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.5, 0.5, 0.5]  # Color gris
                }
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'irregular',
                    'wavelengths': lista_a_string(wave_lengths),
                    'values': lista_a_string(emision_final)
                }
            }
        }
    
    return mi.load_dict(scene_dict)

# %%
# Crear la escena con esferas distribuidas en círculo
print("Creando escena con esferas distribuidas cada 30°...")
# print("Nota: LWIR (8-14 μm) es óptimo para detectar emisión térmica de cuerpos negros")
# print("      ya que la ley de Wien indica que el pico de emisión para temperaturas")
# print("      entre -50°C y 250°C se encuentra en este rango espectral.")
scene = create_circular_spheres_scene(
    sphere_angles,
    temperatures_kelvin,
    distance_from_center, 
    sphere_radius
)

# %%
# Capturar imágenes girando la cámara
ensure_output_dir()
imagenes = []
angulos_captura = []

print("Capturando imágenes con cámara fija en origen...")
for i, camera_angle in enumerate(camera_angles):
    # La cámara está siempre fija en el origen
    camera_x = 0
    camera_y = 0
    camera_z = 0
    
    # Calcular hacia dónde debe mirar la cámara para enfocar la esfera en este ángulo
    angle_rad = math.radians(camera_angle)
    target_x = distance_from_center * math.cos(angle_rad)
    target_y = 0  # Mantener en el plano horizontal
    target_z = distance_from_center * math.sin(angle_rad)
    
    print(f"  Captura {i+1}/{len(camera_angles)}: mirando hacia esfera a {camera_angle}° (target: {target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
    
    # Crear sensor en el origen mirando hacia la esfera objetivo
    sensor = load_fixed_sensor_gaussian(
        camera_x, camera_y, camera_z,
        target_x, target_y, target_z,
        wave_lengths=wave_lengths,
        sigma=sigma,
        fov=fov,
        k=k,
        n_wavelents=n
    )
    
    # Renderizar imagen
    imagen = mi.render(scene, sensor=sensor, spp=spp)
    imagenes.append(imagen)
    angulos_captura.append(camera_angle)

# %%
# Visualizar todas las imágenes capturadas
plt.figure(figsize=(20, 15))

# Determinar el número de filas y columnas para la grilla
n_images = len(imagenes)
cols = 4
rows = (n_images + cols - 1) // cols

# Calcular rango global para normalizar el colormap
banda_central = len(wave_lengths) // 2
valores_globales = []
for imagen in imagenes:
    valores_globales.extend(imagen[:, :, banda_central].numpy().flatten())

vmin_global = np.min(valores_globales)
vmax_global = np.max(valores_globales)

print(f"Rango global del colormap: {vmin_global:.2e} - {vmax_global:.2e}")

for i, (imagen, angulo) in enumerate(zip(imagenes, angulos_captura)):
    plt.subplot(rows, cols, i + 1)
    
    # Mostrar la imagen (banda espectral central) con rango fijo
    im = plt.imshow(imagen[:, :, banda_central], cmap='hot', vmin=vmin_global, vmax=vmax_global)
    plt.colorbar(im)
    plt.title(f'Vista desde {angulo}°')
    plt.axis('off')

plt.suptitle('Esferas emisoras cada 30° - Cámara LWIR fija en origen mirando a cada esfera', fontsize=16)
plt.tight_layout()
plt.savefig('output/circular_spheres_lwir_rotating_camera.png', dpi=400, bbox_inches='tight')
plt.show()

# %%
# Crear una animación mostrando diferentes bandas espectrales
print("Creando visualización de diferentes bandas espectrales...")

# Seleccionar algunas bandas de interés
bandas_interes = [0, len(wave_lengths)//4, len(wave_lengths)//2, 3*len(wave_lengths)//4, -1]
nombres_bandas = [f'{wave_lengths[b]} nm' for b in bandas_interes]

# Calcular rango global para cada banda espectral
rangos_bandas = {}
for banda in bandas_interes:
    valores_banda = []
    for imagen in imagenes:
        valores_banda.extend(imagen[:, :, banda].numpy().flatten())
    rangos_bandas[banda] = (np.min(valores_banda), np.max(valores_banda))

plt.figure(figsize=(25, 12))

for i, banda in enumerate(bandas_interes):
    vmin_banda, vmax_banda = rangos_bandas[banda]
    
    for j, (imagen, angulo) in enumerate(zip(imagenes, angulos_captura)):
        subplot_idx = i * len(imagenes) + j + 1
        plt.subplot(len(bandas_interes), len(imagenes), subplot_idx)
        
        # Usar rango fijo para cada banda
        im = plt.imshow(imagen[:, :, banda], cmap='hot', vmin=vmin_banda, vmax=vmax_banda)
        
        # Solo mostrar títulos en la primera fila
        if i == 0:
            plt.title(f'{angulo}°', fontsize=10)
        
        # Mostrar labels de banda en la primera columna con más información para LWIR
        if j == 0:
            longitud_onda_um = wave_lengths[banda] / 1000  # Convertir a micrómetros
            if i == 0:
                etiqueta = f'{longitud_onda_um:.1f} μm\n(LWIR bajo)'
            elif i == 1:
                etiqueta = f'{longitud_onda_um:.1f} μm\n(LWIR medio-bajo)'
            elif i == 2:
                etiqueta = f'{longitud_onda_um:.1f} μm\n(LWIR medio)'
            elif i == 3:
                etiqueta = f'{longitud_onda_um:.1f} μm\n(LWIR medio-alto)'
            else:
                etiqueta = f'{longitud_onda_um:.1f} μm\n(LWIR alto)'
            plt.ylabel(etiqueta, fontsize=10, rotation=0, labelpad=50, ha='right')
        
        plt.axis('off')

plt.suptitle('Esferas emisoras - Diferentes bandas LWIR (cámara fija)', fontsize=16)
plt.tight_layout()
plt.savefig('output/circular_spheres_lwir_spectral_bands.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Análisis de intensidad total por ángulo de cámara
intensidades_por_angulo = []
for imagen in imagenes:
    intensidad_total = imagen.numpy().sum()
    intensidades_por_angulo.append(intensidad_total)

plt.figure(figsize=(12, 6))
plt.plot(angulos_captura, intensidades_por_angulo, 'o-', linewidth=2, markersize=8)
plt.title('Intensidad total captada vs dirección objetivo')
plt.xlabel('Ángulo de dirección objetivo (grados)')
plt.ylabel('Intensidad total')
plt.grid(True, alpha=0.3)
plt.xticks(angulos_captura)
plt.savefig('output/circular_spheres_lwir_intensity_vs_angle.png', dpi=150, bbox_inches='tight')
plt.show()

# %%


# %%
print("\nAnálisis LWIR completado:")
print(f"- {len(sphere_angles)} esferas posicionadas cada 30°")
print(f"- {len(camera_angles)} imágenes capturadas con cámara fija en origen")
print(f"- Rango espectral: {wave_lengths[0]/1000:.1f} - {wave_lengths[-1]/1000:.1f} μm (LWIR)")
print("- Imágenes guardadas en:")
print("  * output/circular_spheres_lwir_rotating_camera.png")
print("  * output/circular_spheres_lwir_spectral_bands.png") 
print("  * output/circular_spheres_lwir_intensity_vs_angle.png")
