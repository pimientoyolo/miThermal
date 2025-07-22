#!/usr/bin/env python3
"""
Script para visualizar firmas espectrales de materiales ECOSTRESS.

Este script carga y grafica las firmas espectrales de diferentes materiales
de la base de datos ECOSTRESS, mostrando tanto reflectancia como emisividad.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from src.ecostress_utils import (
    load_ecostress_material, 
    plot_ecostress_signatures, 
    plot_single_material
)
from src.visualization_utils import ensure_output_dir

# %%
# Configuración
print("Visualizando firmas espectrales de materiales ECOSTRESS...")

# Directorio de archivos ECOSTRESS
ecostress_dir = "ecostress"

# Encontrar archivos disponibles
available_files = []
if os.path.exists(ecostress_dir):
    for file in os.listdir(ecostress_dir):
        if file.endswith('.spectrum.txt'):
            available_files.append(file)

print(f"Archivos ECOSTRESS encontrados: {len(available_files)}")
for i, file in enumerate(available_files[:10]):  # Mostrar solo los primeros 10
    print(f"  {i+1}. {file}")
if len(available_files) > 10:
    print(f"  ... y {len(available_files) - 10} más")

# %%
# Seleccionar materiales específicos para análisis
materiales_interes = [
    "water.distilledwater.none.liquid.tir.distwatr.jhu.becknic.spectrum.txt",
    "manmade.concrete.pavingconcrete.solid.all.0092uuu_cnc.jhu.becknic.spectrum.txt",
    "manmade.generalconstructionmaterial.brick.solid.all.0097uuubrk.jhu.becknic.spectrum.txt",
    "manmade.generalconstructionmaterial.marble.solid.all.0722uuumbl.jhu.becknic.spectrum.txt",
    "manmade.generalconstructionmaterial.paint.solid.all.0385uuupnt.jhu.becknic.spectrum.txt"
]

# Filtrar solo los archivos que existen
materiales_disponibles = []
for material in materiales_interes:
    file_path = os.path.join(ecostress_dir, material)
    if os.path.exists(file_path):
        materiales_disponibles.append(material)
    else:
        print(f"Advertencia: Archivo no encontrado: {material}")

print(f"\nMateriales seleccionados para análisis: {len(materiales_disponibles)}")

# %%
# Crear directorio de salida
ensure_output_dir()

# %%
# Cargar datos de materiales
print("\nCargando datos de materiales...")

# Longitudes de onda objetivo para LWIR (8-14 μm)
wave_lengths_lwir = np.linspace(4000, 14000, 49)

materials_data = []
for material_file in materiales_disponibles:
    file_path = os.path.join(ecostress_dir, material_file)
    print(f"  Procesando: {material_file}")
    
    try:
        wavelengths_nm, reflectance, emissivity, metadata = load_ecostress_material(
            file_path, target_wavelengths_nm=wave_lengths_lwir
        )
        
        # Nombre corto para la leyenda
        name = metadata.get('Name', material_file.split('.')[0])
        materials_data.append((name, wavelengths_nm, reflectance, emissivity, metadata))
        
        print(f"    ✓ {name}")
        print(f"    Tipo: {metadata.get('Type', 'N/A')}")
        print(f"    Clase: {metadata.get('Class', 'N/A')}")
        
    except Exception as e:
        print(f"    ✗ Error procesando {material_file}: {e}")

print(f"\nMateriales cargados exitosamente: {len(materials_data)}")

# %%
# Visualización 1: Comparación de múltiples materiales (rango completo)
if materials_data:
    print("\nGenerando comparación de materiales en rango completo...")
    
    plot_ecostress_signatures(
        materials_data,
        save_path="output/ecostress_materials_comparison_full.png",
        figsize=(16, 10),
        wavelength_range_um=None,  # Rango completo
        show_reflectance=True,
        show_absorption=True
    )
    
    print("Generando comparación de materiales en LWIR...")
    
    plot_ecostress_signatures(
        materials_data,
        save_path="output/ecostress_materials_comparison_lwir.png",
        figsize=(16, 10),
        wavelength_range_um=(8, 14),  # Solo LWIR
        show_reflectance=True,
        show_absorption=True
    )

# %%
# Visualización 2: Firmas individuales detalladas
print("\nGenerando firmas individuales...")

for i, (name, wavelengths_nm, reflectance, emissivity, metadata) in enumerate(materials_data):
    # Crear nombre de archivo seguro
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_').lower()
    
    # Graficar material individual (rango completo)
    material_file = materiales_disponibles[i]
    file_path = os.path.join(ecostress_dir, material_file)
    
    print(f"  Graficando: {name}")
    plot_single_material(
        file_path,
        save_path=f"output/ecostress_{safe_name}_full_spectrum.png",
        figsize=(12, 8)
    )

# %%
# Visualización 3: Solo emisividad en LWIR para análisis térmico
if materials_data:
    print("\nGenerando comparación de emisividad en LWIR...")
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(materials_data)))
    
    for i, (name, wavelengths_nm, reflectance, emissivity, metadata) in enumerate(materials_data):
        wavelengths_um = wavelengths_nm / 1000.0
        plt.plot(wavelengths_um, emissivity, color=colors[i], linewidth=2, label=name)
    
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel('Emisividad')
    plt.title('Emisividad de Materiales en LWIR (8-14 μm)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(8, 14)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('output/ecostress_emissivity_lwir_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Análisis estadístico
if materials_data:
    print("\nAnálisis estadístico de emisividad en LWIR:")
    print("=" * 60)
    print(f"{'Material':<25} | {'Emisiv. Media':<12} | {'Desv. Std':<10} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)
    
    for name, wavelengths_nm, reflectance, emissivity, metadata in materials_data:
        mean_emis = np.mean(emissivity)
        std_emis = np.std(emissivity)
        min_emis = np.min(emissivity)
        max_emis = np.max(emissivity)
        
        print(f"{name[:24]:<25} | {mean_emis:<12.4f} | {std_emis:<10.4f} | {min_emis:<8.4f} | {max_emis:<8.4f}")
    
    print("=" * 60)

# %%
# Guardar datos procesados
print("\nGuardando datos procesados...")

processed_data = {
    'wavelengths_lwir_nm': wave_lengths_lwir.tolist(),
    'wavelengths_lwir_um': (wave_lengths_lwir / 1000.0).tolist(),
    'materials': []
}

for name, wavelengths_nm, reflectance, emissivity, metadata in materials_data:
    processed_data['materials'].append({
        'name': name,
        'reflectance': reflectance.tolist(),
        'emissivity': emissivity.tolist(),
        'metadata': metadata
    })

with open('output/ecostress_materials_lwir_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("Datos guardados en: output/ecostress_materials_lwir_data.pkl")

# %%
print("\nAnálisis completado!")
print("Archivos generados:")
print("- output/ecostress_materials_comparison_lwir.png")
print("- output/ecostress_emissivity_lwir_comparison.png")
for name, _, _, _, _ in materials_data:
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_').lower()
    print(f"- output/ecostress_{safe_name}_full_spectrum.png")
print("- output/ecostress_materials_lwir_data.pkl")

print(f"\nSe analizaron {len(materials_data)} materiales de ECOSTRESS")
print("Las firmas espectrales muestran las propiedades de reflectancia y emisividad")
print("en función de la longitud de onda, útiles para análisis de sensado remoto térmico.")
