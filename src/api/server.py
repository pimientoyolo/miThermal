"""
Servidor FastAPI para procesamiento de escenas Mitsuba y visualización 3D
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Dict, Optional

# Imports del proyecto
from ..config import get_config, get_output_path
from ..mitsuba_core.scene_parser import MitsubaSceneParser

# Configuración
config = get_config()
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Mitsuba Scene Viewer API",
    description="API para procesar escenas Mitsuba y visualización 3D",
    version="2.0.0"
)

# Configurar CORS para permitir acceso desde el cliente web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio para archivos estáticos (modelos 3D)
STATIC_DIR = get_output_path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Montar directorio estático
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Variable global para almacenar el parser actual
current_parser: Optional[MitsubaSceneParser] = None
current_scene_data: Optional[Dict] = None



