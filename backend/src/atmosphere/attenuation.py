"""
Funciones para manejo de atenuación atmosférica.
"""

import logging
from pathlib import Path
import numpy as np
from fastapi import HTTPException

from src.config import AIR_ATTENUATION_FILE, DEFAULT_ATTENNUATION_DIR

logger = logging.getLogger(__name__)

# Archivos de gases atmosféricos disponibles
ATMOSPHERIC_GAS_FILES = ["air.txt", "H2O.txt", "CO2.txt", "CH4.txt", "O3.txt"]


def get_attenuation(attenuation_file: str = "air.txt") -> tuple[np.ndarray, np.ndarray]:
    """
    Lee el archivo de atenuación completo y retorna dos arrays alineados:
    - wavelengths_nm: longitudes de onda en nanómetros (nm) obtenidas del archivo (columna 0 en µm convertida a nm)
    - sigma_t_neper: coeficientes de extinción en neper (Np), obtenidos de la columna 2 multiplicada por ln(10)/10

    El resultado se ordena de mayor a menor longitud de onda, preservando el pareo (wavelength, sigma_t).

    Args:
        attenuation_file (str): Nombre del archivo de atenuación (sin extensión) en assets/reference_data.

    Returns:
        tuple[np.ndarray, np.ndarray]: (wavelengths_nm_desc, sigma_t_neper_desc)
    """
    try:
        # Construir la ruta del archivo
        file_path = f"{DEFAULT_ATTENNUATION_DIR}/{attenuation_file}"

        # Validar que el archivo existe
        if not Path(file_path).exists():
            available_files = [f.replace('.txt', '') for f in ATMOSPHERIC_GAS_FILES]
            raise HTTPException(
                status_code=404,
                detail=f"Archivo '{attenuation_file}.txt' no encontrado. Disponibles: {available_files}"
            )

        # Cargar los datos del archivo
        trans_array = np.loadtxt(file_path)

        # Columnas esperadas: [wavelength_um, transmittance, attenuation]
        wavelengths_um = trans_array[:, 0].astype(float)
        attenuation_vals = trans_array[:, 2].astype(float)

        # Convertir longitudes de onda a nm
        wavelengths_nm = wavelengths_um * 1000.0

        # Convertir atenuación a neper: multiplicar por ln(10)/10
        # Y convertir de km^-1 a m^-1 (asumiendo escena en metros)
        sigma_t_neper = (attenuation_vals * (np.log(10.0) / 10.0)) / 1000.0

        # Ordenar de menor a mayor por longitud de onda, manteniendo pares
        order = np.argsort(wavelengths_nm)
        wavelengths_nm_desc = wavelengths_nm[order]
        sigma_t_neper_desc = sigma_t_neper[order]

        # Guardar a archivo tab-delimitado (wavelength_nm, sigma_t_neper)
        out_path = Path(AIR_ATTENUATION_FILE)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = np.column_stack((wavelengths_um, sigma_t_neper))
            np.savetxt(out_path, data, delimiter='\t')
        except Exception as save_err:
            # No detener el flujo si falla el guardado; reportar y continuar
            logger.warning(f"No se pudo guardar archivo de atenuación en '{AIR_ATTENUATION_FILE}': {save_err}")

        return wavelengths_nm_desc, sigma_t_neper_desc

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo atenuación de '{attenuation_file}': {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar archivo de atenuación: {e}")


def read_air_attenuation_file() -> tuple[np.ndarray, np.ndarray]:
    """
    Lee el archivo configurado en AIR_ATTENUATION_FILE (dos columnas: wavelength, sigma_t),
    asumiendo que la primera columna está en micrómetros (µm). Convierte las longitudes a nanómetros (nm)
    y retorna ambas columnas ordenadas de mayor a menor longitud de onda.

    Returns:
        tuple[np.ndarray, np.ndarray]: (wavelengths_nm_desc, sigma_t_desc)
    """
    try:
        path = Path(AIR_ATTENUATION_FILE)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"No existe el archivo de atenuación: {path}")

        data = np.loadtxt(path)

        # Manejar casos de una sola fila
        if data.ndim == 1:
            if data.size < 2:
                raise HTTPException(status_code=400, detail="El archivo debe tener al menos dos columnas")
            data = data.reshape(1, -1)

        if data.shape[1] < 2:
            raise HTTPException(status_code=400, detail="El archivo debe tener dos columnas: wavelength_um y sigma_t")

        wavelengths_um = data[:, 0].astype(float)
        sigma_t = data[:, 1].astype(float)

        wavelengths_nm = wavelengths_um * 1000.0

        order = np.argsort(wavelengths_nm)
        wavelengths_nm_desc = wavelengths_nm[order]
        sigma_t_desc = sigma_t[order]

        return wavelengths_nm_desc, sigma_t_desc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leyendo archivo de atenuación: {e}")
        raise HTTPException(status_code=500, detail=f"Error al leer archivo de atenuación: {e}")
