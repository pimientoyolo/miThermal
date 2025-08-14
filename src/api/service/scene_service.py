import logging

from fastapi import UploadFile

from ...mitsuba_core.scene_parser import MitsubaSceneParser
from fastapi import HTTPException
from ...config import get_output_path
import os
import glob
import zipfile

from ...mitsuba_core.object_utils import ObjectUtils
import shutil

object_utils = ObjectUtils()

logger = logging.getLogger(__name__)

out_dir = get_output_path("static")


class SceneService:
    def __init__(self):
        self.logger = logger

    def load_scene(self, file: UploadFile):
        # Verificar si el archivo es un ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un archivo ZIP")

        # Eliminar archivos existentes con extensiones .zip, .obj y .xml
        for ext in ['*.zip', '*.obj', '*.xml', '*.ply', "*.jpg"]:
            for file_path in glob.glob(os.path.join(out_dir, ext)):
                try:
                    os.remove(file_path)
                except OSError as e:
                    self.logger.warning(f"Error al eliminar {file_path}: {e}")
        
        # Eliminar carpeta meshes si existe
        meshes_dir = os.path.join(out_dir, "meshes")
        if os.path.exists(meshes_dir):
            try:
                shutil.rmtree(meshes_dir)
            except OSError as e:
                self.logger.warning(f"Error al eliminar carpeta meshes: {e}")
        
        # guardar archivo zip
        with open(f"{out_dir}/scene.zip", "wb") as buffer:
            buffer.write(file.file.read())

        # descomprimir archivo zip
        with zipfile.ZipFile(f"{out_dir}/scene.zip", 'r') as zip_ref:
            zip_ref.extractall(out_dir)

        # buscar y renombrar .xml
        xml_files = glob.glob(os.path.join(out_dir, "*.xml"))
        if len(xml_files) != 1:
            raise HTTPException(status_code=400, detail="Debe haber exactamente un archivo XML en el ZIP")

        xml_file = xml_files[0]
        new_xml_path = os.path.join(out_dir, "scene.xml")
        os.rename(xml_file, new_xml_path)

        return new_xml_path




        

        
        
