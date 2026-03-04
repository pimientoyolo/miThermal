"""
Funciones para creación de medios participativos (homogéneos) para simulación atmosférica.
"""

import logging
import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def lista_a_string(lista: np.ndarray) -> str:
    """
    Convierte un array numpy a string separado por comas para Mitsuba XML.
    
    Args:
        lista: Array numpy
        
    Returns:
        String con valores separados por comas
    """
    return ", ".join(map(str, lista))


def create_homogeneous_medium(
    wavelengths: np.ndarray, 
    sigma_t: np.ndarray, 
    medium_id: str = "niebla", 
    g_value: float = 0.95
) -> dict:
    """
    Crea un diccionario para un medium homogéneo con coeficiente de extinción espectral.
    
    Args:
        wavelengths (np.ndarray): Array con las longitudes de onda en nanómetros
        sigma_t (np.ndarray): Array con los valores de coeficiente de extinción sigma_t
        medium_id (str): Identificador del medium (por defecto "niebla")
        g_value (float): Parámetro g de la fase Henyey-Greenstein (0.9-0.95 para IR lejano)
        
    Returns:
        dict: Diccionario del medium listo para Mitsuba/XML
        
    Raises:
        HTTPException: Si las longitudes de los arrays no coinciden
    """
    try:
        # Validar que ambos arrays tengan la misma longitud
        if len(wavelengths) != len(sigma_t):
            raise HTTPException(
                status_code=400, 
                detail=f"Las longitudes no coinciden: wavelengths={len(wavelengths)}, sigma_t={len(sigma_t)}"
            )
        
        # Crear el diccionario del medium con estructura para XML según documentación Mitsuba
        medium_dict = {
            "@type": "homogeneous",
            "@id": medium_id,
            "rgb": {
                "@name": "albedo", 
                "@value": "0.0, 0.0, 0.0"
            },
            "spectrum": {
                "@type": "irregular",
                "@name": "sigma_t",
                "string": [
                    {"@name": "wavelengths", "@value": lista_a_string(wavelengths)},
                    {"@name": "values", "@value": lista_a_string(sigma_t)},
                ]
            },
            "phase": {
                "@type": "hg",
                "float": {
                    "@name": "g",
                    "@value": str(g_value)
                }
            }
        }
        
        logger.info(f"Medium homogéneo creado con ID '{medium_id}' y g={g_value}")
        
        return medium_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creando medium homogéneo: {e}")
        raise HTTPException(status_code=500, detail=f"Error al crear medium homogéneo: {e}")
