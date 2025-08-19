import logging
from re import S

import open3d

import trimesh
from pathlib import Path
from fastapi import HTTPException
from src.mitsuba_core.scene_parser import SceneParser
from src.api.dto.suggestDTO import SuggestDTO

class ObjectUtils:
    """
    Utilidades para conversión y manipulación de archivos de objetos 3D
    Soporta conversiones entre PLY, OBJ, STL, y otros formatos
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scene_parser = SceneParser()

    def ply2obj(self, ply_file_path: str) -> str:
        """
        Convierte archivo PLY a OBJ en el mismo directorio
        
        Args:
            ply_file_path: Ruta del archivo PLY origen
            
        Returns:
            Ruta del archivo OBJ creado
        """
        self.valid_exist_file(ply_file_path)

        try:
            # Crear la ruta del archivo OBJ en el mismo directorio que el PLY
            ply_path = Path(ply_file_path)
            obj_file_path = ply_path.with_suffix('.obj')
            
            mesh = open3d.io.read_triangle_mesh(ply_file_path)
            if len(mesh.vertices) == 0:
                raise HTTPException(status_code=400, detail=f"El archivo {ply_file_path} no contiene geometría válida")
            open3d.io.write_triangle_mesh(str(obj_file_path), mesh)
            return str(obj_file_path)
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error converting PLY to OBJ: {e}")
            raise HTTPException(status_code=500, detail=f"Error al convertir el archivo {ply_file_path} a OBJ")

    def obj2ply(self, obj_file_path: str) -> str:
        """
        Convierte archivo OBJ a PLY en el mismo directorio
        
        Args:
            obj_file_path: Ruta del archivo OBJ origen
            
        Returns:
            Ruta del archivo PLY creado
        """
        self.valid_exist_file(obj_file_path)

        try:
            # Crear la ruta del archivo PLY en el mismo directorio que el OBJ
            obj_path = Path(obj_file_path)
            ply_file_path = obj_path.with_suffix('.ply')
            
            mesh = open3d.io.read_triangle_mesh(obj_file_path)
            if len(mesh.vertices) == 0:
                raise HTTPException(status_code=400, detail=f"El archivo {obj_file_path} no contiene geometría válida")
            open3d.io.write_triangle_mesh(str(ply_file_path), mesh)
            return str(ply_file_path)
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error converting OBJ to PLY: {e}")
            raise HTTPException(status_code=500, detail=f"Error al convertir el archivo {obj_file_path} a PLY")

    
    def valid_exist_file(self, path: str):
        """
        Valida si existe un archivo en la ruta especificada
                
        Args:
            path: Ruta del archivo a validar
                    
        Returns:
            True si el archivo existe, False en caso contrario
        """
        try:
            file_path = Path(path)
            if not (file_path.exists() and file_path.is_file()):
                raise HTTPException(status_code=404, detail="Archivo no encontrado")
            return True
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error al validar el archivo: {e}")
            raise HTTPException(status_code=500, detail="Error validando archivo")

    def get_suggested_object(self, scene_path: str) -> list[SuggestDTO]:

        list_suggest: list[SuggestDTO] = []

        scene_dict = self.scene_parser.xml_to_dict(scene_path)
        
        shapes = scene_dict["scene"]["shape"]

        for shape in shapes:
            file = shape["string"]["@value"]
            file_name = Path(file).stem

            sug = SuggestDTO(id=file, suggest=file_name)
            list_suggest.append(sug)

        return list_suggest