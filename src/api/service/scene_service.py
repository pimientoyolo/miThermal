import logging

from fastapi import UploadFile

from ...mitsuba_core.scene_parser import MitsubaSceneParser
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class SceneService:
    def __init__(self):
        self.logger = logger

    def load_scene(self, file: UploadFile):
        # Verificar si el archivo es un ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un archivo ZIP")

        if file.content_type != 'application/zip':
            raise HTTPException(status_code=400, detail="Tipo de contenido inválido. Se esperaba application/zip")
