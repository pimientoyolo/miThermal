#!/usr/bin/env python3
"""
Script para comparar los resultados de los tres análisis de distancias.

Este script carga los resultados generados por los análisis:
- dragon_distances.py
- point_source_distances.py
- area_emitter_distances.py

Y crea gráficas comparativas para visualizar las diferencias entre
los tres tipos de emisores.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from src.visualization_utils import ensure_output_dir

# Configuración de la comparación
parser = argparse.ArgumentParser(description='Comparar resultados de análisis de emisión')
parser.add_argument('--tipo', choices=['intensidad', 'espectro', 'error'], default='todos', 
                   help='Tipo de comparación: intensidad, espectro, error o todos')
args = parser.parse_args()

ensure_output_dir('output')
ensure_output_dir('output/comparison')

# Cargar los resultados desde los archivos pickle
resultados = {}
emisores = {
    'dragon': 'Emisor Dragón (Área)',
    'point_emitter': 'Emisor Puntual',
    'area_emitter': 'Emisor Esférico (Área)'
}

# Intentar cargar los resultados de cada emisor
for emisor_key, emisor_nombre in emisores.items():
    archivo = f'output/{emisor_key}_results.pkl'
    try:
        if os.path.exists(archivo):
            with open(archivo, 'rb') as f:
                resultados[emisor_key] = pickle.load(f)
                print(f"Resultados cargados para {emisor_nombre}")
        else:
            print(f"No se encontró el archivo de resultados para {emisor_nombre}")
    except Exception as e:
        print(f"Error al cargar los resultados de {emisor_nombre}: {e}")

# Verificar si se cargaron resultados
if not resultados:
    print("No se encontraron resultados. Ejecute primero los scripts de análisis.")
    exit(1)

# Colores para los gráficos
colores = {
    'dragon': 'firebrick',
    'point_emitter': 'royalblue',
    'area_emitter': 'forestgreen',
    'area': 'darkorange',
    'inversa': 'black'
}

# Nombres descriptivos
nombres = {
    'dragon': 'Dragón (área)',
    'point': 'Fuente puntual',
    'area': 'Esfera emisora (área)',
    'inversa': 'Ley 1/r²'
}

# Función para comparar la caída de intensidad
def comparar_caida_intensidad():
    plt.figure(figsize=(12, 8))
    
    # Graficar la ley teórica del cuadrado inverso en escala log-log
    x_teorico = np.linspace(1, 12, 100)
    y_teorico = 1 / (x_teorico ** 2)
    plt.plot(x_teorico, y_teorico, 'k--', linewidth=2, alpha=0.7, label='Ley 1/r²')
    
    # Graficar los datos de cada emisor
    for emisor_key, data in resultados.items():
        if 'intensidad_relativa' in data and 'distancias' in data:
            plt.plot(data['distancias'], data['intensidad_relativa'], 'o-', 
                    color=colores.get(emisor_key, 'gray'),
                    linewidth=2, markersize=8, 
                    label=emisores[emisor_key])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.title('Comparación de caída de intensidad entre emisores', fontsize=16)
    plt.xlabel('Factor de distancia', fontsize=14)
    plt.ylabel('Intensidad relativa (normalizada)', fontsize=14)
    plt.legend(fontsize=12)
    
    plt.savefig('output/comparison/intensity_falloff_comparison.png')
    plt.tight_layout()
    plt.show()
    
    print("Comparación de intensidad guardada en 'output/comparison/intensity_falloff_comparison.png'")

# Función para comparar los espectros
def comparar_espectros():
    plt.figure(figsize=(12, 8))
    
    # Crear subplots para comparar espectros a diferentes distancias
    # Elegimos la primera distancia para la comparación
    distancia_idx = 0 # La primera distancia (1x)
    
    # Graficar cada espectro
    for emisor_key, data in resultados.items():
        if 'pixeles_normalizados' in data and 'wave_lengths' in data:
            wave_lengths = np.array(data['wave_lengths'])
            spectrum = data['pixeles_normalizados'][distancia_idx]
            plt.plot(wave_lengths, spectrum, '-', 
                    color=colores.get(emisor_key, 'gray'),
                    linewidth=2, 
                    label=emisores[emisor_key])
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Comparación de espectros normalizados a distancia 1x', fontsize=16)
    plt.xlabel('Longitud de onda (nm)', fontsize=14)
    plt.ylabel('Emisión relativa (normalizada)', fontsize=14)
    plt.legend(fontsize=12)
    
    plt.savefig('output/comparison/spectrum_comparison.png')
    plt.tight_layout()
    plt.show()
    
    print("Comparación de espectros guardada en 'output/comparison/spectrum_comparison.png'")

# Función para comparar el error respecto a la ley del cuadrado inverso
def comparar_error_cuadrado():
    plt.figure(figsize=(12, 8))
    
    # Configurar ancho de barras
    num_emisores = len(resultados)
    ancho_grupo = 0.8  # Ancho total del grupo de barras
    ancho_barra = ancho_grupo / num_emisores
    
    # Obtener todas las distancias y asegurarnos de que sean las mismas para todos
    for emisor_key, data in resultados.items():
        if 'distancias' in data:
            distancias = data['distancias']
            break
    
    ind = np.arange(len(distancias))  # Posiciones de las barras
    
    # Graficar barras para cada emisor
    i = 0
    for emisor_key, data in resultados.items():
        if 'error_porcentual' in data:
            offset = ancho_barra * (i - num_emisores / 2 + 0.5)
            plt.bar(ind + offset, data['error_porcentual'], ancho_barra, 
                   label=emisores[emisor_key],
                   color=colores.get(emisor_key, 'gray'))
            i += 1
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(ind, [f'{d}x' for d in distancias])
    plt.title('Error porcentual respecto a la ley del cuadrado inverso', fontsize=16)
    plt.xlabel('Factor de distancia', fontsize=14)
    plt.ylabel('Error (%)', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.savefig('output/comparison/inverse_square_error_comparison.png')
    plt.tight_layout()
    plt.show()
    
    print("Comparación de error guardada en 'output/comparison/inverse_square_error_comparison.png'")

# Ejecutar las comparaciones según el tipo solicitado
if args.tipo == 'intensidad' or args.tipo == 'todos':
    comparar_caida_intensidad()

if args.tipo == 'espectro' or args.tipo == 'todos':
    comparar_espectros()

if args.tipo == 'error' or args.tipo == 'todos':
    comparar_error_cuadrado()
    
    print("Comparación de espectros guardada en 'output/comparison/spectrum_comparison.png'")

print("\nNota: Este script utiliza datos simulados para la comparación.")
print("Para obtener una comparación precisa, los scripts de análisis deberían")
print("guardar sus resultados en archivos que este script pueda cargar.")
