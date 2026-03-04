"""
Servidor FastAPI para procesamiento de escenas Mitsuba y visualización 3D
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

# Imports del proyecto
from ..config import get_config, get_output_path

# Routers
from src.api.controller.scene_controller import scene_router
from src.api.controller.obj_controller import obj_router
from src.api.controller.render_controller import render_router
from src.api.controller.config_controller import config_router
from src.api.controller.spectral_controller import router as spectral_router


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


app.include_router(
    scene_router
)

app.include_router(
    obj_router
)

app.include_router(
    render_router
)

app.include_router(
    config_router
)

app.include_router(
    spectral_router
)
