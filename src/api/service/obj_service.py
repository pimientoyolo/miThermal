import logging
from src.mitsuba_core.object_utils import ObjectUtils
import os
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

class ObjService:
    def __init__(self):
        self.logger = logger
        self.object_utils = ObjectUtils()

    def get_object(self, dir :str) -> FileResponse:
        if not os.path.exists(dir):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {dir}")

        file_extension = os.path.splitext(dir)[1].lower()
        
        if file_extension == '.obj':
            return self.get_obj_response(dir)
        
        elif file_extension == '.ply':
            path_obj = self.object_utils.ply2obj(dir)
            return self.get_obj_response(path_obj)

        else:
            raise HTTPException(status_code=400, detail="Solo se soporta archivos .obj y .ply")

    def get_obj_response(self, path: str) -> FileResponse:
        def generate():
            with open(path, 'rb') as file:
                while chunk := file.read(8192):
                    yield chunk

        return StreamingResponse(
            generate(),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(path)}"}
        )
