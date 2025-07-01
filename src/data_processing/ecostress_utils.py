"""
Utilidades para cargar y procesar datos espectrales de ECOSTRESS.

Este módulo contiene funciones para leer archivos de reflectancia de ECOSTRESS,
convertir unidades de micrómetros a nanómetros, y calcular absorción a partir
de reflectancia.
"""

import numpy as np
import os
from typing import Tuple, Optional
import re

# Grosor efectivo del material en metros (por defecto 1 mm)
DEFAULT_THICKNESS_M = 0.005


def parse_ecostress_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Lee un archivo de datos espectrales de ECOSTRESS y extrae los datos de reflectancia.
    
    Args:
        file_path (str): Ruta al archivo de ECOSTRESS
        
    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: 
            - wavelengths_um: Longitudes de onda en micrómetros
            - reflectance_percent: Reflectancia en porcentaje
            - metadata: Diccionario con metadatos del archivo
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    metadata = {}
    wavelengths = []
    reflectance = []
    data_started = False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Saltar líneas vacías
            if not line:
                continue
            
            # Parsear metadatos
            if ':' in line and not data_started:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
                continue
            
            # Detectar inicio de datos (primera línea que empieza con número)
            if re.match(r'^\s*\d+\.?\d*', line):
                data_started = True
            
            # Procesar datos espectrales
            if data_started:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        wl = float(parts[0])
                        refl = float(parts[1])
                        wavelengths.append(wl)
                        reflectance.append(refl)
                except ValueError:
                    # Saltar líneas que no se pueden convertir a float
                    continue
    
    if not wavelengths:
        raise ValueError(f"No se encontraron datos espectrales válidos en {file_path}")
    
    return np.array(wavelengths), np.array(reflectance), metadata


def ecostress_to_nanometers(wavelengths_um: np.ndarray, 
                           reflectance_percent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte longitudes de onda de micrómetros a nanómetros y reflectancia de porcentaje a fracción.
    
    Args:
        wavelengths_um (np.ndarray): Longitudes de onda en micrómetros
        reflectance_percent (np.ndarray): Reflectancia en porcentaje (0-100)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - wavelengths_nm: Longitudes de onda en nanómetros
            - reflectance_fraction: Reflectancia como fracción (0-1)
    """
    wavelengths_nm = wavelengths_um * 1000  # Convertir μm a nm
    reflectance_fraction = reflectance_percent / 100.0  # Convertir % a fracción
    
    # Asegurar que la reflectancia esté en el rango válido [0, 1]
    reflectance_fraction = np.clip(reflectance_fraction, 0.0, 1.0)
    
    return wavelengths_nm, reflectance_fraction


def reflectance_to_absorptance(reflectance: np.ndarray) -> np.ndarray:
    """
    Calcula la absortancia (absorción) a partir de la reflectancia.
    
    Asume que la transmitancia es cero (material opaco), por lo que:
    Absorptancia = 1 - Reflectancia
    
    Args:
        reflectance (np.ndarray): Reflectancia como fracción (0-1)
        
    Returns:
        np.ndarray: Absortancia como fracción (0-1)
    """
    # Asegurar que la reflectancia esté en el rango válido
    reflectance = np.clip(reflectance, 0.0, 1.0)
    
    # Para materiales opacos: A = 1 - R (donde T = 0)
    absorptance = 1.0 - reflectance
    
    return absorptance


def reflectance_to_emissivity(reflectance: np.ndarray) -> np.ndarray:
    """
    Calcula la emisividad a partir de la reflectancia usando la ley de Kirchhoff.
    
    Para materiales en equilibrio térmico: Emisividad = Absortancia = 1 - Reflectancia
    
    Args:
        reflectance (np.ndarray): Reflectancia como fracción (0-1)
        
    Returns:
        np.ndarray: Emisividad como fracción (0-1)
    """
    return reflectance_to_absorptance(reflectance)


def load_ecostress_material(file_path: str, 
                           target_wavelengths_nm: Optional[np.ndarray] = None,
                           thickness_m: float = DEFAULT_THICKNESS_M) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Carga un material de ECOSTRESS y devuelve reflectancia, emisividad, transmitancia y sigma_t.
    
    Args:
        file_path (str): Ruta al archivo de ECOSTRESS
        target_wavelengths_nm (np.ndarray, opcional): Longitudes de onda objetivo en nm para interpolación
        thickness_m (float): Grosor efectivo del material en metros para calcular sigma_t
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            - wavelengths_nm: Longitudes de onda en nanómetros
            - reflectance: Reflectancia como fracción (0-1)
            - emissivity: Emisividad como fracción (0-1)
            - sigma_t: Coeficiente de extinción en [1/m]
            - metadata: Metadatos del archivo
    """
    # Cargar datos originales
    wavelengths_um, reflectance_percent, metadata = parse_ecostress_file(file_path)
    
    # Convertir unidades
    wavelengths_nm, reflectance = ecostress_to_nanometers(wavelengths_um, reflectance_percent)
    
    # Calcular emisividad y sigma_t
    emissivity = reflectance_to_emissivity(reflectance)
    sigma_t = reflectance_to_sigma_t(reflectance, thickness_m)
    
    # Interpolación si se especifican longitudes de onda objetivo
    if target_wavelengths_nm is not None:
        reflectance_interp = np.interp(target_wavelengths_nm, wavelengths_nm, reflectance)
        emissivity_interp = np.interp(target_wavelengths_nm, wavelengths_nm, emissivity)
        sigma_t_interp = np.interp(target_wavelengths_nm, wavelengths_nm, sigma_t)
        return target_wavelengths_nm, reflectance_interp, emissivity_interp, sigma_t_interp, metadata
    
    return wavelengths_nm, reflectance, emissivity, sigma_t, metadata


def interpolate_to_lwir(wavelengths_nm: np.ndarray, 
                       values: np.ndarray, 
                       lwir_range: Tuple[float, float] = (8000, 14000),
                       num_points: int = 49) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpola datos espectrales al rango LWIR.
    
    Args:
        wavelengths_nm (np.ndarray): Longitudes de onda originales en nm
        values (np.ndarray): Valores espectrales (reflectancia, emisividad, etc.)
        lwir_range (Tuple[float, float]): Rango LWIR en nm (por defecto 8-14 μm)
        num_points (int): Número de puntos en el rango LWIR
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - lwir_wavelengths: Longitudes de onda LWIR en nm
            - lwir_values: Valores interpolados al rango LWIR
    """
    lwir_wavelengths = np.linspace(lwir_range[0], lwir_range[1], num_points)
    lwir_values = np.interp(lwir_wavelengths, wavelengths_nm, values)
    
    return lwir_wavelengths, lwir_values


def get_material_name_from_filename(file_path: str) -> str:
    """
    Extrae un nombre de material legible del nombre del archivo de ECOSTRESS.
    
    Args:
        file_path (str): Ruta al archivo de ECOSTRESS
        
    Returns:
        str: Nombre del material extraído
    """
    filename = os.path.basename(file_path)
    
    # Remover extensión
    name_without_ext = os.path.splitext(filename)[0]
    
    # Dividir por puntos y tomar las partes relevantes
    parts = name_without_ext.split('.')
    
    if len(parts) >= 3:
        # Formato típico: manmade.concrete.pavingconcrete.solid.all.0092uuu_cnc.jhu.becknic.spectrum
        material_class = parts[1]  # concrete, metal, etc.
        material_subclass = parts[2] if len(parts) > 2 else ""
        
        # Crear nombre legible
        if material_subclass and material_subclass != material_class:
            return f"{material_class}_{material_subclass}"
        else:
            return material_class
    
    return filename


def example_usage():
    """Función de ejemplo que muestra cómo usar las utilidades de ECOSTRESS."""
    
    # Ejemplo de uso con un archivo de concreto
    file_path = "ecostress/manmade.concrete.pavingconcrete.solid.all.0092uuu_cnc.jhu.becknic.spectrum.txt"
    
    try:
        # Cargar material completo
        wavelengths_nm, reflectance, emissivity, sigma_t, metadata = load_ecostress_material(file_path)
        
        print(f"Material cargado: {get_material_name_from_filename(file_path)}")
        print(f"Rango de longitudes de onda: {wavelengths_nm[0]:.1f} - {wavelengths_nm[-1]:.1f} nm")
        print(f"Número de puntos espectrales: {len(wavelengths_nm)}")
        print(f"Reflectancia media: {np.mean(reflectance):.3f}")
        print(f"Emisividad media: {np.mean(emissivity):.3f}")
        print(f"σₜ medio: {np.mean(sigma_t):.2e} m⁻¹")
        
        # Validar conservación de energía
        is_valid, total = validate_energy_conservation(reflectance)
        print(f"\nConservación de energía válida: {is_valid}")
        if not is_valid:
            print(f"Desviación máxima: {np.max(np.abs(total - 1.0)):.6f}")
        
        # Calcular coeficientes adicionales
        sigma_a = reflectance_to_absorption_coefficient(reflectance)
        sigma_s = reflectance_to_scattering_coefficient(reflectance, absorption_fraction=0.8)
        albedo = calculate_single_scattering_albedo(sigma_a, sigma_s)
        
        print("\nCoeficientes espectrales (valores medios):")
        print(f"σₐ medio: {np.mean(sigma_a):.2e} m⁻¹")
        print(f"σₛ medio: {np.mean(sigma_s):.2e} m⁻¹")
        print(f"Albedo medio: {np.mean(albedo):.3f}")
        
        # Interpolar al rango LWIR
        lwir_wavelengths, lwir_sigma_t = interpolate_to_lwir(wavelengths_nm, sigma_t)
        print("\nDatos LWIR interpolados:")
        print(f"Rango LWIR: {lwir_wavelengths[0]:.0f} - {lwir_wavelengths[-1]:.0f} nm")
        print(f"σₜ medio en LWIR: {np.mean(lwir_sigma_t):.2e} m⁻¹")
        
        # Mostrar algunos metadatos
        print("\nMetadatos:")
        for key in ['Name', 'Type', 'Class', 'Description']:
            if key in metadata:
                print(f"{key}: {metadata[key]}")
        
        # Ejemplo de exportación a Mitsuba
        print("\nEjemplo de exportación a Mitsuba:")
        medium = create_physically_realistic_medium(
            lwir_wavelengths, 
            np.interp(lwir_wavelengths, wavelengths_nm, reflectance),
            absorption_fraction=0.8,
            scattering_anisotropy=0.1
        )
        
        print("Medio físicamente realista creado:")
        print(f"  Tipo: {medium['type']}")
        print(f"  σₜ rango: {medium['_metadata']['sigma_t_range']}")
        print(f"  Albedo rango: {medium['_metadata']['albedo_range']}")
        print(f"  Función de fase: {medium['phase']['type']}")
                
    except Exception as e:
        print(f"Error en el ejemplo: {e}")


def analyze_material_statistics(materials_data: list, 
                              wavelength_range_nm: Optional[Tuple[float, float]] = None) -> dict:
    """
    Analiza estadísticas de múltiples materiales ECOSTRESS.
    
    Args:
        materials_data (list): Lista de materiales cargados
        wavelength_range_nm (tuple, opcional): Rango de longitudes de onda para análisis
        
    Returns:
        dict: Estadísticas resumidas de todos los materiales
    """
    if not materials_data:
        return {}
    
    stats = {
        "num_materials": len(materials_data),
        "reflectance": {"mean": [], "std": [], "min": [], "max": []},
        "sigma_t": {"mean": [], "std": [], "min": [], "max": []},
        "absorptance": {"mean": [], "std": [], "min": [], "max": []},
        "material_names": []
    }
    
    for name, wavelengths_nm, reflectance, emissivity, sigma_t, metadata in materials_data:
        # Filtrar por rango si se especifica
        if wavelength_range_nm is not None:
            mask = (wavelengths_nm >= wavelength_range_nm[0]) & (wavelengths_nm <= wavelength_range_nm[1])
            reflectance_analysis = reflectance[mask]
            sigma_t_analysis = sigma_t[mask]
        else:
            reflectance_analysis = reflectance
            sigma_t_analysis = sigma_t
        
        absorptance_analysis = reflectance_to_absorptance(reflectance_analysis)
        
        # Calcular estadísticas
        stats["material_names"].append(name)
        
        for prop_name, values in [("reflectance", reflectance_analysis), 
                                 ("sigma_t", sigma_t_analysis),
                                 ("absorptance", absorptance_analysis)]:
            stats[prop_name]["mean"].append(np.mean(values))
            stats[prop_name]["std"].append(np.std(values))
            stats[prop_name]["min"].append(np.min(values))
            stats[prop_name]["max"].append(np.max(values))
    
    # Convertir listas a arrays para estadísticas globales
    for prop_name in ["reflectance", "sigma_t", "absorptance"]:
        for stat_name in ["mean", "std", "min", "max"]:
            stats[prop_name][stat_name] = np.array(stats[prop_name][stat_name])
    
    # Añadir estadísticas globales
    for prop_name in ["reflectance", "sigma_t", "absorptance"]:
        stats[f"{prop_name}_global"] = {
            "overall_mean": np.mean(stats[prop_name]["mean"]),
            "overall_std": np.std(stats[prop_name]["mean"]),
            "range_min": np.min(stats[prop_name]["min"]),
            "range_max": np.max(stats[prop_name]["max"])
        }
    
    return stats


def print_material_statistics(stats: dict) -> None:
    """
    Imprime estadísticas de materiales de forma legible.
    
    Args:
        stats (dict): Estadísticas de analyze_material_statistics
    """
    print(f"\n=== ESTADÍSTICAS DE {stats['num_materials']} MATERIALES ECOSTRESS ===")
    
    print("\nMateriales analizados:")
    for i, name in enumerate(stats["material_names"]):
        print(f"  {i+1}. {name}")
    
    print("\nEstadísticas Globales:")
    properties = [
        ("Reflectancia", "reflectance", ""),
        ("Absorptancia", "absorptance", ""),
        ("σₜ", "sigma_t", " m⁻¹")
    ]
    
    for prop_display, prop_name, units in properties:
        global_stats = stats[f"{prop_name}_global"]
        print(f"\n{prop_display}:")
        print(f"  Media global: {global_stats['overall_mean']:.4f}{units}")
        print(f"  Desv. estándar: {global_stats['overall_std']:.4f}{units}")
        print(f"  Rango: {global_stats['range_min']:.4f} - {global_stats['range_max']:.4f}{units}")


def find_materials_by_property(materials_data: list, 
                              property_name: str,
                              threshold: float,
                              comparison: str = "greater") -> list:
    """
    Encuentra materiales que cumplen un criterio específico.
    
    Args:
        materials_data (list): Lista de materiales cargados
        property_name (str): Propiedad a evaluar ("reflectance", "sigma_t", "absorptance")
        threshold (float): Valor umbral
        comparison (str): Tipo de comparación ("greater", "less", "equal")
        
    Returns:
        list: Lista de materiales que cumplen el criterio
    """
    matching_materials = []
    
    for material_data in materials_data:
        name, wavelengths_nm, reflectance, emissivity, sigma_t, metadata = material_data
        
        # Seleccionar propiedad
        if property_name == "reflectance":
            values = reflectance
        elif property_name == "sigma_t":
            values = sigma_t
        elif property_name == "absorptance":
            values = reflectance_to_absorptance(reflectance)
        else:
            continue
        
        # Calcular valor medio de la propiedad
        mean_value = np.mean(values)
        
        # Aplicar criterio
        matches = False
        if comparison == "greater" and mean_value > threshold:
            matches = True
        elif comparison == "less" and mean_value < threshold:
            matches = True
        elif comparison == "equal" and abs(mean_value - threshold) < 0.1 * threshold:
            matches = True
        
        if matches:
            matching_materials.append((name, mean_value, material_data))
    
    # Ordenar por valor de la propiedad
    matching_materials.sort(key=lambda x: x[1], reverse=(comparison == "greater"))
    
    return matching_materials


def plot_ecostress_signatures(materials_data: list, 
                              save_path: Optional[str] = None,
                              figsize: tuple = (15, 12),
                              wavelength_range_um: Optional[tuple] = None,
                              show_reflectance: bool = True,
                              show_absorption: bool = True,
                              show_sigma_t: bool = True) -> None:
    """
    Grafica las firmas espectrales de múltiples materiales de ECOSTRESS.
    
    Args:
        materials_data (list): Lista de tuplas (nombre, wavelengths_nm, reflectance, emissivity, sigma_t, metadata)
        save_path (str, opcional): Ruta para guardar la figura
        figsize (tuple): Tamaño de la figura
        wavelength_range_um (tuple, opcional): Rango de longitudes de onda en micrómetros (min, max)
        show_reflectance (bool): Si mostrar reflectancia
        show_absorption (bool): Si mostrar absorción
        show_sigma_t (bool): Si mostrar coeficiente de extinción
    """
    import matplotlib.pyplot as plt
    
    n_plots = sum([show_reflectance, show_absorption, show_sigma_t])
    if n_plots == 0:
        print("Error: Debe mostrar al menos una propiedad espectral")
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Colores para diferentes materiales
    colors = plt.cm.tab10(np.linspace(0, 1, len(materials_data)))
    
    plot_idx = 0
    
    # Graficar reflectancia
    if show_reflectance:
        ax = axes[plot_idx]
        for i, (name, wavelengths_nm, reflectance, emissivity, sigma_t, metadata) in enumerate(materials_data):
            # Convertir a micrómetros para el eje X
            wavelengths_um = wavelengths_nm / 1000.0
            
            # Filtrar por rango si se especifica
            if wavelength_range_um is not None:
                mask = (wavelengths_um >= wavelength_range_um[0]) & (wavelengths_um <= wavelength_range_um[1])
                wavelengths_um = wavelengths_um[mask]
                reflectance_plot = reflectance[mask]
            else:
                reflectance_plot = reflectance
            
            ax.plot(wavelengths_um, reflectance_plot * 100, 
                   color=colors[i], linewidth=2, label=name)
        
        ax.set_xlabel('Longitud de onda (μm)')
        ax.set_ylabel('Reflectancia (%)')
        ax.set_title('Firmas Espectrales - Reflectancia')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)
        plot_idx += 1
    
    # Graficar absorción
    if show_absorption:
        ax = axes[plot_idx]
        for i, (name, wavelengths_nm, reflectance, emissivity, sigma_t, metadata) in enumerate(materials_data):
            # Convertir a micrómetros para el eje X
            wavelengths_um = wavelengths_nm / 1000.0
            
            # Calcular absorción a partir de reflectancia
            absorption = reflectance_to_absorptance(reflectance)
            
            # Filtrar por rango si se especifica
            if wavelength_range_um is not None:
                mask = (wavelengths_um >= wavelength_range_um[0]) & (wavelengths_um <= wavelength_range_um[1])
                wavelengths_um = wavelengths_um[mask]
                absorption_plot = absorption[mask]
            else:
                absorption_plot = absorption
            
            ax.plot(wavelengths_um, absorption_plot, 
                       color=colors[i], linewidth=2, label=name)
        
        ax.set_xlabel('Longitud de onda (μm)')
        ax.set_ylabel('Absorción (escala log)')
        ax.set_title('Firmas Espectrales - Absorción')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(1e-4, 1)
        plot_idx += 1
    
    # Graficar sigma_t
    if show_sigma_t:
        ax = axes[plot_idx]
        for i, (name, wavelengths_nm, reflectance, emissivity, sigma_t, metadata) in enumerate(materials_data):
            # Convertir a micrómetros para el eje X
            wavelengths_um = wavelengths_nm / 1000.0
            
            # Filtrar por rango si se especifica
            if wavelength_range_um is not None:
                mask = (wavelengths_um >= wavelength_range_um[0]) & (wavelengths_um <= wavelength_range_um[1])
                wavelengths_um = wavelengths_um[mask]
                sigma_t_plot = sigma_t[mask]
            else:
                sigma_t_plot = sigma_t
            
            ax.plot(wavelengths_um, sigma_t_plot, 
                       color=colors[i], linewidth=2, label=name)
        
        ax.set_xlabel('Longitud de onda (μm)')
        ax.set_ylabel('σₜ (1/m,g)')
        ax.set_title('Coeficiente de Extinción σₜ')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()

def plot_single_material(file_path: str, 
                        target_wavelengths_nm: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None,
                        figsize: tuple = (12, 12),
                        wavelength_range_um: Optional[tuple] = None,
                        thickness_m: float = DEFAULT_THICKNESS_M) -> None:
    """
    Grafica la firma espectral de un solo material de ECOSTRESS.
    
    Args:
        file_path (str): Ruta al archivo de ECOSTRESS
        target_wavelengths_nm (np.ndarray, opcional): Longitudes de onda objetivo para interpolación
        save_path (str, opcional): Ruta para guardar la figura
        figsize (tuple): Tamaño de la figura
        wavelength_range_um (tuple, opcional): Rango de longitudes de onda en micrómetros
        thickness_m (float): Grosor efectivo del material en metros
    """
    import matplotlib.pyplot as plt
    
    # Cargar material
    wavelengths_nm, reflectance, emissivity, sigma_t, metadata = load_ecostress_material(
        file_path, target_wavelengths_nm, thickness_m
    )
    
    # Calcular absorción y transmitancia
    absorption = reflectance_to_absorptance(reflectance)
    transmittance = reflectance_to_transmittance(reflectance)
    
    # Convertir a micrómetros
    wavelengths_um = wavelengths_nm / 1000.0
    
    # Filtrar por rango si se especifica
    if wavelength_range_um is not None:
        mask = (wavelengths_um >= wavelength_range_um[0]) & (wavelengths_um <= wavelength_range_um[1])
        wavelengths_um = wavelengths_um[mask]
        reflectance = reflectance[mask]
        absorption = absorption[mask]
        transmittance = transmittance[mask]
        sigma_t = sigma_t[mask]
    
    # Crear figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Reflectancia y Transmitancia
    ax1.plot(wavelengths_um, reflectance * 100, 'b-', linewidth=2, label='Reflectancia')
    ax1.plot(wavelengths_um, transmittance * 100, 'g-', linewidth=2, label='Transmitancia')
    ax1.set_ylabel('Porcentaje (%)')
    ax1.set_title(f"Firma Espectral: {metadata.get('Name', 'Material desconocido')}")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    ax1.legend()
    
    # Absorción (escala logarítmica)
    ax2.semilogy(wavelengths_um, absorption, 'r-', linewidth=2)
    ax2.set_ylabel('Absorción (escala log)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-4, 1)
    
    # Sigma_t (escala logarítmica)
    ax3.semilogy(wavelengths_um, sigma_t, 'purple', linewidth=2)
    ax3.set_xlabel('Longitud de onda (μm)')
    ax3.set_ylabel('σₜ (1/m,g)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'Coeficiente de Extinción (grosor = {thickness_m*1000:.1f} mm)')
    
    # Información del material
    info_text = []
    for key in ['Type', 'Class', 'Subclass']:
        if key in metadata and metadata[key]:
            info_text.append(f"{key}: {metadata[key]}")
    
    if info_text:
        ax1.text(0.02, 0.98, '\n'.join(info_text), 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()

def reflectance_to_sigma_t(reflectance: np.ndarray, thickness_m: float = DEFAULT_THICKNESS_M) -> np.ndarray:
    """
    Calcula el coeficiente de extinción σₜ(λ) a partir de la reflectancia.
    
    Utiliza la relación física:
    T(λ) = exp(-σₜ(λ) × d)
    
    Por lo tanto:
    σₜ(λ) = -ln(T(λ)) / d
    
    donde T(λ) se calcula como T = 1 - R para materiales con absorción despreciable
    en el cálculo de transmitancia superficial.

    Args:
        reflectance (np.ndarray): Reflectancia como fracción [0, 1)
        thickness_m (float): Grosor efectivo del material en metros

    Returns:
        np.ndarray: σₜ(λ) en unidades [1/m]
    """
    # Asegurar valores válidos y evitar log(0)
    reflectance = np.clip(reflectance, 0.0, 0.9999)
    
    # Calcular transmitancia: T = 1 - R (asume absorción superficial despreciable)
    transmittance = 1.0 - reflectance
    
    # Evitar valores muy pequeños que causen problemas numéricos
    transmittance = np.maximum(transmittance, 1e-10)
    
    # Calcular σₜ usando la ley de Beer-Lambert
    sigma_t = -np.log(transmittance) / thickness_m
    
    return sigma_t


def reflectance_to_absorption_coefficient(reflectance: np.ndarray, thickness_m: float = DEFAULT_THICKNESS_M) -> np.ndarray:
    """
    Calcula el coeficiente de absorción σₐ(λ) a partir de la reflectancia.
    
    Para materiales donde la absorción domina sobre el scattering:
    σₐ ≈ σₜ = -ln(T) / d = -ln(1 - R) / d

    Args:
        reflectance (np.ndarray): Reflectancia como fracción [0, 1)
        thickness_m (float): Grosor efectivo del material en metros

    Returns:
        np.ndarray: σₐ(λ) en unidades [1/m]
    """
    return reflectance_to_sigma_t(reflectance, thickness_m)


def reflectance_to_scattering_coefficient(reflectance: np.ndarray, 
                                        absorption_fraction: float = 0.9,
                                        thickness_m: float = DEFAULT_THICKNESS_M) -> np.ndarray:
    """
    Calcula el coeficiente de scattering σₛ(λ) a partir de la reflectancia.
    
    Asume que σₜ = σₐ + σₛ y que σₐ = absorption_fraction × σₜ
    Por lo tanto: σₛ = (1 - absorption_fraction) × σₜ

    Args:
        reflectance (np.ndarray): Reflectancia como fracción [0, 1)
        absorption_fraction (float): Fracción de extinción debida a absorción (0-1)
        thickness_m (float): Grosor efectivo del material en metros

    Returns:
        np.ndarray: σₛ(λ) en unidades [1/m]
    """
    sigma_t = reflectance_to_sigma_t(reflectance, thickness_m)
    sigma_s = (1.0 - absorption_fraction) * sigma_t
    return sigma_s


def calculate_single_scattering_albedo(sigma_a: np.ndarray, sigma_s: np.ndarray) -> np.ndarray:
    """
    Calcula el albedo de single scattering ω(λ) = σₛ/(σₐ + σₛ) = σₛ/σₜ.

    Args:
        sigma_a (np.ndarray): Coeficiente de absorción [1/m]
        sigma_s (np.ndarray): Coeficiente de scattering [1/m]

    Returns:
        np.ndarray: Albedo de single scattering [0, 1]
    """
    sigma_t = sigma_a + sigma_s
    # Evitar división por cero
    albedo = np.where(sigma_t > 1e-10, sigma_s / sigma_t, 0.0)
    return np.clip(albedo, 0.0, 1.0)


def validate_energy_conservation(reflectance: np.ndarray, 
                               transmittance: Optional[np.ndarray] = None,
                               absorptance: Optional[np.ndarray] = None,
                               tolerance: float = 1e-6) -> Tuple[bool, np.ndarray]:
    """
    Valida la conservación de energía: R + T + A = 1.
    
    Args:
        reflectance (np.ndarray): Reflectancia [0, 1]
        transmittance (np.ndarray, opcional): Transmitancia [0, 1]
        absorptance (np.ndarray, opcional): Absorptancia [0, 1]
        tolerance (float): Tolerancia para la validación
        
    Returns:
        Tuple[bool, np.ndarray]: (es_válido, suma_total)
    """
    if transmittance is None:
        transmittance = reflectance_to_transmittance(reflectance)
    
    if absorptance is None:
        absorptance = reflectance_to_absorptance(reflectance)
    
    total = reflectance + transmittance + absorptance
    is_valid = np.all(np.abs(total - 1.0) <= tolerance)
    
    return is_valid, total


def reflectance_to_transmittance(reflectance: np.ndarray) -> np.ndarray:
    """
    Calcula la transmitancia a partir de la reflectancia.
    
    Asume que T = 1 - R (sin absorción interna significativa)
    
    Args:
        reflectance (np.ndarray): Reflectancia como fracción (0-1)
        
    Returns:
        np.ndarray: Transmitancia como fracción (0-1)
    """
    # Asegurar que la reflectancia esté en el rango válido
    reflectance = np.clip(reflectance, 0.0, 0.9999)
    
    transmittance = 1.0 - reflectance
    
    return transmittance


def export_mitsuba_spectrum(wavelengths_nm: np.ndarray, values: np.ndarray, 
                           spectrum_type: str = "sigma_t") -> dict:
    """
    Exporta datos espectrales en formato compatible con Mitsuba.
    
    Args:
        wavelengths_nm (np.ndarray): Longitudes de onda en nanómetros
        values (np.ndarray): Valores espectrales (sigma_t, reflectancia, etc.)
        spectrum_type (str): Tipo de espectro para documentación
        
    Returns:
        dict: Diccionario compatible con Mitsuba
    """
    return {
        "type": "irregular",
        "wavelengths": ", ".join([str(int(w)) for w in wavelengths_nm]),
        "values": ", ".join([f"{v:.6e}" for v in values]),
        "_comment": f"Spectrum type: {spectrum_type}, units: {get_spectrum_units(spectrum_type)}"
    }


def get_spectrum_units(spectrum_type: str) -> str:
    """
    Retorna las unidades para diferentes tipos de espectros.
    
    Args:
        spectrum_type (str): Tipo de espectro
        
    Returns:
        str: Unidades correspondientes
    """
    units_map = {
        "sigma_t": "1/m",
        "reflectance": "dimensionless",
        "emissivity": "dimensionless",
        "transmittance": "dimensionless",
        "absorption": "dimensionless"
    }
    return units_map.get(spectrum_type, "unknown")


def create_mitsuba_medium(sigma_t_dict: dict, albedo: float = 0.0, 
                         phase_type: str = "isotropic", g: float = 0.0) -> dict:
    """
    Crea un diccionario de medio para Mitsuba usando sigma_t calculado.
    
    Args:
        sigma_t_dict (dict): Diccionario de espectro sigma_t de export_mitsuba_spectrum
        albedo (float): Albedo del medio (0.0 = solo absorción, 1.0 = solo scattering)
        phase_type (str): Tipo de función de fase ("isotropic" o "hg")
        g (float): Parámetro de anisotropía para Henyey-Greenstein (solo si phase_type="hg")
        
    Returns:
        dict: Diccionario de medio compatible con Mitsuba
    """
    medium_dict = {
        "type": "homogeneous",
        "sigma_t": {
            "type": "constvolume",
            "value": sigma_t_dict
        },
        "albedo": {
            "type": "constvolume", 
            "value": {"type": "uniform", "value": albedo}
        }
    }
    
    if phase_type == "hg":
        medium_dict["phase"] = {"type": "hg", "g": g}
    else:
        medium_dict["phase"] = {"type": "isotropic"}
    
    return medium_dict


def create_physically_realistic_medium(wavelengths_nm: np.ndarray, 
                                     reflectance: np.ndarray,
                                     thickness_m: float = DEFAULT_THICKNESS_M,
                                     absorption_fraction: float = 0.9,
                                     scattering_anisotropy: float = 0.0) -> dict:
    """
    Crea un medio físicamente realista para Mitsuba basado en datos de reflectancia.
    
    Args:
        wavelengths_nm (np.ndarray): Longitudes de onda en nm
        reflectance (np.ndarray): Reflectancia medida [0, 1]
        thickness_m (float): Grosor efectivo del material
        absorption_fraction (float): Fracción de σₜ debida a absorción [0, 1]
        scattering_anisotropy (float): Parámetro g de Henyey-Greenstein [-1, 1]
        
    Returns:
        dict: Medio completo para Mitsuba con todos los coeficientes espectrales
    """
    # Calcular coeficientes espectrales
    sigma_t = reflectance_to_sigma_t(reflectance, thickness_m)
    sigma_a = absorption_fraction * sigma_t
    sigma_s = (1.0 - absorption_fraction) * sigma_t
    albedo = calculate_single_scattering_albedo(sigma_a, sigma_s)
    
    # Crear espectros para Mitsuba
    sigma_t_spectrum = export_mitsuba_spectrum(wavelengths_nm, sigma_t, "sigma_t")
    albedo_spectrum = export_mitsuba_spectrum(wavelengths_nm, albedo, "albedo")
    
    # Configurar función de fase
    if abs(scattering_anisotropy) > 1e-6:
        phase_func = {"type": "hg", "g": scattering_anisotropy}
    else:
        phase_func = {"type": "isotropic"}
    
    medium = {
        "type": "homogeneous",
        "sigma_t": sigma_t_spectrum,
        "albedo": albedo_spectrum,
        "phase": phase_func,
        "_metadata": {
            "source": "ECOSTRESS reflectance data",
            "thickness_m": thickness_m,
            "absorption_fraction": absorption_fraction,
            "sigma_t_range": [float(np.min(sigma_t)), float(np.max(sigma_t))],
            "albedo_range": [float(np.min(albedo)), float(np.max(albedo))]
        }
    }
    
    return medium


def export_mitsuba_json(medium_dict: dict, file_path: str) -> None:
    """
    Exporta un diccionario de medio a un archivo JSON compatible con Mitsuba.
    
    Args:
        medium_dict (dict): Diccionario del medio de Mitsuba
        file_path (str): Ruta donde guardar el archivo JSON
    """
    import json
    
    # Convertir numpy arrays a listas para serialización JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return obj
    
    # Crear una copia del diccionario para no modificar el original
    exportable_dict = json.loads(json.dumps(medium_dict, default=convert_numpy))
    
    with open(file_path, 'w') as f:
        json.dump(exportable_dict, f, indent=2)
    
    print(f"Medio exportado a: {file_path}")


def compare_material_properties(materials_data: list, 
                              property_name: str = "sigma_t",
                              wavelength_range_nm: Optional[Tuple[float, float]] = None,
                              save_path: Optional[str] = None) -> None:
    """
    Compara una propiedad específica entre múltiples materiales.
    
    Args:
        materials_data (list): Lista de materiales cargados
        property_name (str): Propiedad a comparar ("sigma_t", "reflectance", "absorptance")
        wavelength_range_nm (tuple, opcional): Rango de longitudes de onda en nm
        save_path (str, opcional): Ruta para guardar la figura
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    for i, (name, wavelengths_nm, reflectance, emissivity, sigma_t, metadata) in enumerate(materials_data):
        # Seleccionar la propiedad a graficar
        if property_name == "sigma_t":
            values = sigma_t
            ylabel = "σₜ (1/m)"
            use_log = True
        elif property_name == "reflectance":
            values = reflectance * 100
            ylabel = "Reflectancia (%)"
            use_log = False
        elif property_name == "absorptance":
            values = reflectance_to_absorptance(reflectance) * 100
            ylabel = "Absorptancia (%)"
            use_log = False
        elif property_name == "emissivity":
            values = emissivity * 100
            ylabel = "Emisividad (%)"
            use_log = False
        else:
            raise ValueError(f"Propiedad no reconocida: {property_name}")
        
        # Convertir a micrómetros
        wavelengths_um = wavelengths_nm / 1000.0
        
        # Filtrar por rango si se especifica
        if wavelength_range_nm is not None:
            mask = (wavelengths_nm >= wavelength_range_nm[0]) & (wavelengths_nm <= wavelength_range_nm[1])
            wavelengths_um = wavelengths_um[mask]
            values = values[mask]
        
        # Graficar
        if use_log:
            plt.semilogy(wavelengths_um, values, linewidth=2, label=name)
        else:
            plt.plot(wavelengths_um, values, linewidth=2, label=name)
    
    plt.xlabel('Longitud de onda (μm)')
    plt.ylabel(ylabel)
    plt.title(f'Comparación de {property_name.capitalize()} entre Materiales')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparación guardada en: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    example_usage()
