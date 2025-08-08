"""
Parser para archivos XML de Mitsuba - Extracción de objetos 3D
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MitsubaSceneObject:
    """Clase para representar un objeto 3D en una escena Mitsuba"""
    
    def __init__(self, obj_id: str, obj_type: str, filename: str, 
                 transform: Optional[Dict] = None, bsdf: Optional[Dict] = None, 
                 emitter: Optional[Dict] = None):
        self.id = obj_id
        self.type = obj_type
        self.filename = filename
        self.transform = transform or {}
        self.bsdf = bsdf or {}
        self.emitter = emitter or {}
        self.absolute_path = None
    
    def to_dict(self) -> Dict:
        """Convierte el objeto a diccionario para serialización JSON"""
        return {
            'id': self.id,
            'type': self.type,
            'filename': self.filename,
            'absolute_path': str(self.absolute_path) if self.absolute_path else None,
            'transform': self.transform,
            'bsdf': self.bsdf,
            'emitter': self.emitter,
            'has_material': bool(self.bsdf),
            'has_emission': bool(self.emitter)
        }

class MitsubaSceneParser:
    """Parser para archivos XML de escenas Mitsuba"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.objects: List[MitsubaSceneObject] = []
        self.scene_info = {}
    
    def parse_xml_file(self, xml_path: str) -> Dict:
        """
        Parsea un archivo XML de Mitsuba y extrae todos los objetos 3D
        
        Args:
            xml_path (str): Ruta al archivo XML de la escena
            
        Returns:
            Dict: Información de la escena y objetos encontrados
        """
        try:
            xml_path = Path(xml_path)
            if not xml_path.exists():
                raise FileNotFoundError(f"Archivo XML no encontrado: {xml_path}")
            
            # Establecer directorio base relativo al XML
            self.base_path = xml_path.parent
            
            # Parsear XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Resetear objetos
            self.objects = []
            
            # Extraer información general de la escena
            self.scene_info = {
                'xml_path': str(xml_path),
                'base_path': str(self.base_path),
                'integrator': self._extract_integrator(root),
                'sensor': self._extract_sensor(root),
                'emitters': self._extract_emitters(root)
            }
            
            # Extraer objetos 3D
            self._extract_3d_objects(root)
            
            # Resolver rutas absolutas
            self._resolve_object_paths()
            
            return {
                'scene_info': self.scene_info,
                'objects': [obj.to_dict() for obj in self.objects],
                'total_objects': len(self.objects),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error parseando XML: {e}")
            return {
                'scene_info': {},
                'objects': [],
                'total_objects': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_3d_objects(self, root: ET.Element):
        """Extrae todos los objetos 3D del XML"""
        
        # Buscar elementos shape (objetos geométricos)
        for shape in root.findall('.//shape'):
            obj = self._parse_shape_element(shape)
            if obj:
                self.objects.append(obj)
        
        # Buscar elementos con geometría explícita (obj, ply, etc.)
        for element in root.findall('.//*[@type]'):
            if element.get('type') in ['obj', 'ply', 'serialized']:
                obj = self._parse_geometry_element(element)
                if obj:
                    self.objects.append(obj)
    
    def _parse_shape_element(self, shape: ET.Element) -> Optional[MitsubaSceneObject]:
        """Parsea un elemento shape"""
        try:
            shape_type = shape.get('type', 'unknown')
            shape_id = shape.get('id', f'shape_{len(self.objects)}')
            
            # Buscar archivo de geometría
            filename = None
            string_elem = shape.find('./string[@name="filename"]')
            if string_elem is not None:
                filename = string_elem.get('value')
            
            if not filename:
                return None
            
            # Extraer transform
            transform = self._extract_transform(shape)
            
            # Extraer BSDF
            bsdf = self._extract_bsdf(shape)
            
            # Extraer emitter
            emitter = self._extract_emitter_from_shape(shape)
            
            return MitsubaSceneObject(
                obj_id=shape_id,
                obj_type=shape_type,
                filename=filename,
                transform=transform,
                bsdf=bsdf,
                emitter=emitter
            )
            
        except Exception as e:
            logger.warning(f"Error parseando shape: {e}")
            return None
    
    def _parse_geometry_element(self, element: ET.Element) -> Optional[MitsubaSceneObject]:
        """Parsea un elemento de geometría directa"""
        try:
            obj_type = element.get('type')
            obj_id = element.get('id', f'{obj_type}_{len(self.objects)}')
            
            # Buscar filename
            filename = None
            string_elem = element.find('./string[@name="filename"]')
            if string_elem is not None:
                filename = string_elem.get('value')
            
            if not filename:
                return None
            
            return MitsubaSceneObject(
                obj_id=obj_id,
                obj_type=obj_type,
                filename=filename
            )
            
        except Exception as e:
            logger.warning(f"Error parseando elemento de geometría: {e}")
            return None
    
    def _extract_transform(self, element: ET.Element) -> Dict:
        """Extrae información de transformación"""
        transform = {}
        
        transform_elem = element.find('./transform')
        if transform_elem is not None:
            # Buscar diferentes tipos de transformaciones
            for child in transform_elem:
                if child.tag == 'translate':
                    transform['translate'] = self._parse_vector(child)
                elif child.tag == 'rotate':
                    transform['rotate'] = self._parse_rotation(child)
                elif child.tag == 'scale':
                    transform['scale'] = self._parse_vector_or_scalar(child)
                elif child.tag == 'matrix':
                    transform['matrix'] = self._parse_matrix(child)
        
        return transform
    
    def _extract_bsdf(self, element: ET.Element) -> Dict:
        """Extrae información del BSDF (material)"""
        bsdf = {}
        
        bsdf_elem = element.find('./bsdf')
        if bsdf_elem is not None:
            bsdf['type'] = bsdf_elem.get('type', 'unknown')
            bsdf['id'] = bsdf_elem.get('id')
            
            # Extraer propiedades del material
            for prop in bsdf_elem.findall('./*'):
                if prop.tag == 'rgb':
                    name = prop.get('name')
                    value = prop.get('value')
                    if name and value:
                        bsdf[name] = self._parse_rgb(value)
                elif prop.tag == 'float':
                    name = prop.get('name')
                    value = prop.get('value')
                    if name and value:
                        bsdf[name] = float(value)
                elif prop.tag == 'string':
                    name = prop.get('name')
                    value = prop.get('value')
                    if name and value:
                        bsdf[name] = value
        
        return bsdf
    
    def _extract_emitter_from_shape(self, element: ET.Element) -> Dict:
        """Extrae información del emisor asociado a un shape"""
        emitter = {}
        
        emitter_elem = element.find('./emitter')
        if emitter_elem is not None:
            emitter['type'] = emitter_elem.get('type', 'unknown')
            emitter['id'] = emitter_elem.get('id')
            
            # Extraer propiedades del emisor
            for prop in emitter_elem.findall('./*'):
                if prop.tag == 'rgb':
                    name = prop.get('name')
                    value = prop.get('value')
                    if name and value:
                        emitter[name] = self._parse_rgb(value)
                elif prop.tag == 'spectrum':
                    name = prop.get('name')
                    emitter[name] = self._extract_spectrum(prop)
        
        return emitter
    
    def _extract_integrator(self, root: ET.Element) -> Dict:
        """Extrae información del integrador"""
        integrator_elem = root.find('./integrator')
        if integrator_elem is not None:
            return {
                'type': integrator_elem.get('type'),
                'id': integrator_elem.get('id')
            }
        return {}
    
    def _extract_sensor(self, root: ET.Element) -> Dict:
        """Extrae información del sensor/cámara"""
        sensor_elem = root.find('./sensor')
        if sensor_elem is not None:
            return {
                'type': sensor_elem.get('type'),
                'id': sensor_elem.get('id')
            }
        return {}
    
    def _extract_emitters(self, root: ET.Element) -> List[Dict]:
        """Extrae información de emisores globales"""
        emitters = []
        for emitter_elem in root.findall('./emitter'):
            emitters.append({
                'type': emitter_elem.get('type'),
                'id': emitter_elem.get('id')
            })
        return emitters
    
    def _parse_vector(self, element: ET.Element) -> List[float]:
        """Parsea un vector 3D"""
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        z = float(element.get('z', 0))
        return [x, y, z]
    
    def _parse_rotation(self, element: ET.Element) -> Dict:
        """Parsea una rotación"""
        return {
            'x': float(element.get('x', 0)),
            'y': float(element.get('y', 0)),
            'z': float(element.get('z', 0)),
            'angle': float(element.get('angle', 0))
        }
    
    def _parse_vector_or_scalar(self, element: ET.Element) -> any:
        """Parsea un vector o escalar"""
        if 'value' in element.attrib:
            return float(element.get('value'))
        else:
            return self._parse_vector(element)
    
    def _parse_matrix(self, element: ET.Element) -> List[List[float]]:
        """Parsea una matriz 4x4"""
        value = element.get('value', '')
        numbers = [float(x) for x in value.split()]
        # Convertir a matriz 4x4
        matrix = []
        for i in range(4):
            row = numbers[i*4:(i+1)*4]
            matrix.append(row)
        return matrix
    
    def _parse_rgb(self, value: str) -> List[float]:
        """Parsea un valor RGB"""
        return [float(x) for x in value.split()]
    
    def _extract_spectrum(self, element: ET.Element) -> Dict:
        """Extrae información espectral"""
        spectrum = {
            'type': element.get('type', 'unknown')
        }
        
        if element.get('filename'):
            spectrum['filename'] = element.get('filename')
        elif element.get('value'):
            spectrum['value'] = element.get('value')
        
        return spectrum
    
    def _resolve_object_paths(self):
        """Resuelve las rutas absolutas de los archivos de objetos"""
        for obj in self.objects:
            if obj.filename:
                # Intentar resolver ruta relativa al XML
                relative_path = self.base_path / obj.filename
                if relative_path.exists():
                    obj.absolute_path = relative_path.resolve()
                else:
                    # Buscar en directorios comunes
                    search_paths = [
                        self.base_path / 'objects',
                        self.base_path / 'meshes',
                        self.base_path / 'models',
                        self.base_path.parent / 'objects',
                        self.base_path.parent / 'meshes',
                        self.base_path.parent / 'models'
                    ]
                    
                    for search_path in search_paths:
                        potential_path = search_path / Path(obj.filename).name
                        if potential_path.exists():
                            obj.absolute_path = potential_path.resolve()
                            break
                    
                    if not obj.absolute_path:
                        logger.warning(f"No se encontró el archivo: {obj.filename}")

def parse_mitsuba_scene(xml_path: str) -> Dict:
    """
    Función de conveniencia para parsear una escena Mitsuba
    
    Args:
        xml_path (str): Ruta al archivo XML
        
    Returns:
        Dict: Información parseada de la escena
    """
    parser = MitsubaSceneParser()
    return parser.parse_xml_file(xml_path)
