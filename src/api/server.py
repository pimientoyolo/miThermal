"""
Servidor FastAPI para procesamiento de escenas Mitsuba y visualización 3D
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import tempfile
import shutil
from typing import List, Dict, Optional
import uvicorn

# Imports del proyecto
from ..config import get_config, get_output_path
from ..mitsuba_core.scene_parser import MitsubaSceneParser

# Configuración
config = get_config()
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Mitsuba Scene Viewer API",
    description="API para procesar escenas XML de Mitsuba y servir modelos 3D",
    version="1.0.0"
)

# Configurar CORS para permitir acceso desde el cliente web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
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

class SceneManager:
    """Gestor de escenas para mantener el estado del servidor"""
    
    def __init__(self):
        self.parser = None
        self.scene_data = None
        self.uploaded_files = {}
    
    def load_scene(self, xml_path: str) -> Dict:
        """Carga una nueva escena XML"""
        try:
            self.parser = MitsubaSceneParser()
            self.scene_data = self.parser.parse_xml_file(xml_path)
            return self.scene_data
        except Exception as e:
            logger.error(f"Error cargando escena: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_objects(self) -> List[Dict]:
        """Obtiene la lista de objetos de la escena actual"""
        if not self.scene_data:
            raise HTTPException(status_code=404, detail="No hay escena cargada")
        return self.scene_data.get('objects', [])
    
    def get_object_by_id(self, object_id: str) -> Dict:
        """Obtiene un objeto específico por ID"""
        objects = self.get_objects()
        for obj in objects:
            if obj['id'] == object_id:
                return obj
        raise HTTPException(status_code=404, detail=f"Objeto no encontrado: {object_id}")

# Instancia global del gestor de escenas
scene_manager = SceneManager()

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Mitsuba Scene Viewer API",
        "version": "1.0.0",
        "endpoints": {
            "upload_scene": "/upload-scene",
            "load_scene": "/load-scene",
            "get_objects": "/objects",
            "get_object": "/objects/{object_id}",
            "download_object": "/download/{object_id}",
            "scene_info": "/scene-info"
        }
    }

@app.post("/upload-scene")
async def upload_scene(file: UploadFile = File(...)):
    """
    Sube y procesa un archivo XML de escena Mitsuba
    """
    try:
        # Validar tipo de archivo
        if not file.filename.endswith('.xml'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos XML")
        
        # Guardar archivo temporal
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / file.filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Procesar escena
        scene_data = scene_manager.load_scene(str(temp_path))
        
        # Limpiar archivo temporal
        shutil.rmtree(temp_dir)
        
        return {
            "status": "success",
            "message": f"Escena {file.filename} procesada exitosamente",
            "scene_data": scene_data
        }
        
    except Exception as e:
        logger.error(f"Error procesando archivo subido: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-scene")
async def load_scene(scene_path: str):
    """
    Carga una escena XML desde una ruta del servidor
    """
    try:
        xml_path = Path(scene_path)
        if not xml_path.exists():
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {scene_path}")
        
        scene_data = scene_manager.load_scene(scene_path)
        
        return {
            "status": "success",
            "message": f"Escena cargada desde {scene_path}",
            "scene_data": scene_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cargando escena: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scene-info")
async def get_scene_info():
    """
    Obtiene información general de la escena actual
    """
    if not scene_manager.scene_data:
        raise HTTPException(status_code=404, detail="No hay escena cargada")
    
    return {
        "scene_info": scene_manager.scene_data.get('scene_info', {}),
        "total_objects": scene_manager.scene_data.get('total_objects', 0),
        "status": scene_manager.scene_data.get('status', 'unknown')
    }

@app.get("/objects")
async def get_objects():
    """
    Obtiene la lista completa de objetos 3D de la escena
    """
    try:
        objects = scene_manager.get_objects()
        
        return {
            "objects": objects,
            "total": len(objects),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo objetos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/objects/{object_id}")
async def get_object(object_id: str):
    """
    Obtiene información detallada de un objeto específico
    """
    try:
        obj = scene_manager.get_object_by_id(object_id)
        
        return {
            "object": obj,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo objeto {object_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{object_id}")
async def download_object(object_id: str):
    """
    Descarga el archivo de modelo 3D de un objeto específico
    """
    try:
        obj = scene_manager.get_object_by_id(object_id)
        
        if not obj.get('absolute_path'):
            raise HTTPException(status_code=404, detail=f"Archivo de modelo no encontrado para objeto: {object_id}")
        
        file_path = Path(obj['absolute_path'])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Archivo no existe: {file_path}")
        
        return FileResponse(
            path=str(file_path),
            filename=f"{object_id}_{file_path.name}",
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando objeto {object_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/objects/{object_id}/update")
async def update_object_properties(object_id: str, properties: Dict):
    """
    Actualiza las propiedades de un objeto (preparado para futuras funcionalidades)
    """
    try:
        obj = scene_manager.get_object_by_id(object_id)
        
        # Por ahora solo retornamos los datos actuales
        # En el futuro aquí se implementaría la lógica de actualización
        
        return {
            "message": f"Actualización preparada para objeto {object_id}",
            "current_properties": obj,
            "requested_updates": properties,
            "status": "pending_implementation"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando objeto {object_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Endpoint de salud del servidor
    """
    return {
        "status": "healthy",
        "scene_loaded": scene_manager.scene_data is not None,
        "total_objects": len(scene_manager.get_objects()) if scene_manager.scene_data else 0
    }

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """
    Ejecuta el servidor FastAPI
    
    Args:
        host (str): Host del servidor
        port (int): Puerto del servidor
        reload (bool): Recarga automática en desarrollo
    """
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
