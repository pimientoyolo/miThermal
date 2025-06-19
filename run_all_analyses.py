#!/usr/bin/env python3
"""
Script para ejecutar los tres análisis de distancias secuencialmente.

Este script ejecuta los análisis de emisión para:
1. Modelo de dragón (emisor de área)
2. Fuente puntual
3. Emisor esférico (emisor de área)

La ejecución secuencial permite comparar los resultados entre los diferentes 
emisores y estudiar cómo la intensidad lumínica varía con la distancia.
"""
import os
import time
import subprocess
import argparse

# Configurar el parser de argumentos
parser = argparse.ArgumentParser(description='Ejecutar análisis de emisión a diferentes distancias')
parser.add_argument('--desarrollo', action='store_true', help='Ejecutar en modo desarrollo con menos muestras')
parser.add_argument('--solo', choices=['dragon', 'point', 'area'], help='Ejecutar solo uno de los análisis')
args = parser.parse_args()

# Asegurarse de que el directorio de salida existe
if not os.path.exists('output'):
    os.makedirs('output')
    print("Directorio 'output' creado para almacenar resultados")

# Definir los scripts a ejecutar
scripts = {
    'dragon': 'dragon_distances.py',
    'point': 'point_source_distances.py',
    'area': 'area_emitter_distances.py'
}

# Función para ejecutar un script
def ejecutar_script(nombre, archivo):
    print(f"\n{'=' * 80}")
    print(f"Iniciando análisis de {nombre}...")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    try:
        # Ejecutar el script como un módulo Python
        result = subprocess.run(['python', archivo], check=True)
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\n{'=' * 80}")
            print(f"Análisis de {nombre} completado en {elapsed:.2f} segundos")
            print(f"{'=' * 80}\n")
            return True
    except subprocess.CalledProcessError as e:
        print(f"\nError ejecutando {nombre}: {e}")
    except Exception as e:
        print(f"\nError inesperado ejecutando {nombre}: {e}")
    
    return False

# Ejecutar los scripts según los argumentos
if args.solo:
    if args.solo in scripts:
        ejecutar_script(args.solo, scripts[args.solo])
    else:
        print(f"Error: Análisis '{args.solo}' no reconocido")
else:
    # Ejecutar todos los scripts en orden
    for nombre, archivo in scripts.items():
        success = ejecutar_script(nombre, archivo)
        if not success:
            print(f"Advertencia: El análisis de {nombre} falló o fue interrumpido")
            respuesta = input("¿Desea continuar con el siguiente análisis? (s/n): ")
            if respuesta.lower() != 's':
                print("Ejecución interrumpida por el usuario")
                break
    
    print("\nTodos los análisis han sido completados.")
    print("Los resultados se encuentran en el directorio 'output'.")

# Ejecutar la comparación automáticamente
print("\nEjecutando comparación de resultados...")
print("=" * 80)

try:
    # Ejecutar el script de comparación con todas las opciones
    result = subprocess.run(['python', 'compare_results.py', '--tipo', 'todos'], check=True)
    if result.returncode == 0:
        print("Comparación de resultados completada exitosamente.")
    else:
        print("La comparación de resultados no se completó correctamente.")
        print("Puede ejecutarla manualmente con: python compare_results.py --tipo todos")
except Exception as e:
    print(f"Error al ejecutar la comparación de resultados: {e}")
    print("Puede ejecutarla manualmente con los siguientes comandos:")
    print("  - Para todas las comparaciones: 'python compare_results.py --tipo todos'")
    print("  - Para comparación de intensidad: 'python compare_results.py --tipo intensidad'")
    print("  - Para comparación de espectros: 'python compare_results.py --tipo espectro'")
    print("  - Para comparación de errores: 'python compare_results.py --tipo error'")
