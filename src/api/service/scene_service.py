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

        # Eliminar todo el contenido del directorio
        if os.path.exists(out_dir):
            for item in os.listdir(out_dir):
                item_path = os.path.join(out_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)    
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except OSError as e:
                    self.logger.warning(f"Error al eliminar {item_path}: {e}")
        
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




        

        
        
