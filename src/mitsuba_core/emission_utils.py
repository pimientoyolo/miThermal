"""
Utilidades para el cálculo de emisiones, radiancia y carga de datos espectrales.
"""
import numpy as np
from scipy import constants as const

def blackbody_radiance_nm(wavelengths_nm, temperature):
    """
    Compute spectral radiance B(λ, T) of a black body
    using scipy constants.

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

def load_material_data():
    """
    Carga los datos de emisividad de los materiales.
    
    Returns:
        tuple: (nombres de materiales, librería de materiales)
    """
    dataBaseName = np.load('data/matName_FullDatabase.npy', allow_pickle=True).item()["matName"]
    dataBaseName = dataBaseName.squeeze() 
    dataBaseName = np.hstack(dataBaseName)  # lista de nombres de los materiales

    dataBaseLib = np.load('data/matLib_FullDatabase.npy', allow_pickle=True).item()["matLib"] 
    dataBaseLib = dataBaseLib[::-1, :]  # Reverso el orden de la base de datos
    
    return dataBaseName, dataBaseLib

def get_material_signature(material_name, dataBaseName, dataBaseLib):
    """
    Obtiene la firma espectral de un material específico.
    
    Args:
        material_name (str): Nombre del material
        dataBaseName (np.ndarray): Base de datos de nombres de materiales
        dataBaseLib (np.ndarray): Base de datos de firmas espectrales
        
    Returns:
        np.ndarray: Firma espectral del material
    """
    indice_material = np.where(dataBaseName == material_name)[0][0]
    firma = dataBaseLib[:, indice_material]
    return firma

def load_spd_data(file_path):
    """
    Carga datos de emisión espectral desde un archivo .spd.
    
    Args:
        file_path (str): Ruta al archivo .spd
        
    Returns:
        tuple: (wavelengths, values)
    """
    spd_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('//'):  # Saltar comentarios
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                spd_data.append((float(parts[0]), float(parts[1])))

    # Extraer wavelengths y valores de la emisión
    wavelengths = [entry[0] for entry in spd_data]
    values = [entry[1] for entry in spd_data]
    
    return wavelengths, values
