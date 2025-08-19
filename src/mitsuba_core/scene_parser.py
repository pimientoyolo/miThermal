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

    def dict_to_xml(self, scene_dict: Dict) -> str:
        return xmltodict.unparse(scene_dict, pretty=True)
