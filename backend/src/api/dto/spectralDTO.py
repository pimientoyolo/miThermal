"""
DTOs para datos espectrales listos para plotear en el frontend.
"""

from pydantic import BaseModel
from typing import List, Optional


class SpectralDataResponse(BaseModel):
    """Respuesta con datos espectrales línea recta (x, y)."""
    wavelengths: List[float]  # nm
    values: List[float]        # emisividad, transmitancia, etc.
    unit: str                  # "Emisividad (0-1)", "Transmitancia (%)", etc.
    label: str                 # Descripción del dato
    title: Optional[str] = None  # Título del gráfico

class ObjectSpectralDataResponse(BaseModel):
    """Respuesta unificada con emisividad y reflectancia de un objeto."""
    wavelengths: List[float]
    emissivity: List[float]
    reflectance: List[float]
    object_id: str
    label: str
    title: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "wavelengths": [8000, 8100, 8200],
                "values": [0.85, 0.87, 0.89],
                "unit": "Emisividad (0-1)",
                "label": "Espectro de emisividad: concrete.solid",
                "title": "Emisividad vs Longitud de Onda"
            }
        }


class AtmosphericDataResponse(BaseModel):
    """Respuesta con datos atmosféricos (atenuación, transmitancia)."""
    wavelengths: List[float]  # nm
    attenuation: List[float]  # dB/km o neper/m
    transmittance: Optional[List[float]] = None  # % (0-100)
    gas: str  # "air", "H2O", "CO2", etc.
    label: str
    title: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "wavelengths": [8000, 8100, 8200],
                "attenuation": [0.5, 0.55, 0.60],
                "transmittance": [95.0, 94.5, 94.0],
                "gas": "air",
                "label": "Atenuación atmosférica",
                "title": "Atenuación del aire vs Longitud de Onda"
            }
        }


class BlackbodyDataResponse(BaseModel):
    """Datos de radiancia de cuerpo negro a diferentes temperaturas."""
    wavelengths: List[float]  # nm
    temperatures: List[int]  # K
    radiances: List[List[float]]  # [temp_idx][wavelength_idx]
    title: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "wavelengths": [8000, 8100, 8200],
                "temperatures": [300, 400, 500],
                "radiances": [
                    [1.5, 1.52, 1.54],
                    [5.0, 5.1, 5.2],
                    [12.0, 12.3, 12.6]
                ],
                "title": "Radiancia de Planck vs Temperatura"
            }
        }
