"""
Parser para archivos XML de Mitsuba - Extracción de objetos 3D
"""
from typing import Dict
import logging
import xmltodict

logger = logging.getLogger(__name__)

class SceneParser:

    def xml_to_dict(self, path_xml : str) -> Dict:

     with open(path_xml) as f:
            scene_dict = xmltodict.parse(f.read())
            return scene_dict

    def dict_to_xml(self, scene_dict: str) -> str:
        return xmltodict.unparse(scene_dict, pretty=True)

    def save_dict_as_xml(self, scene_dict: Dict, output_path: str) -> None:
        """
        Guarda un diccionario como archivo XML
        
        Args:
            scene_dict: Diccionario a convertir a XML
            output_path: Ruta donde guardar el archivo XML
        """
        xml_content = xmltodict.unparse(scene_dict, pretty=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        logger.info(f"XML guardado en: {output_path}")

    
