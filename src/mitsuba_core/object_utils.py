import logging
from re import S

import open3d

import trimesh
import json
import numpy as np
from pathlib import Path
from fastapi import HTTPException
from src.mitsuba_core.scene_parser import SceneParser
from src.api.dto.suggestDTO import SuggestDTO
from scipy import constants as const

class ObjectUtils:
    """
    Utilidades para conversión y manipulación de archivos de objetos 3D
    Soporta conversiones entre PLY, OBJ, STL, y otros formatos
    """
    
    # Rutas de los archivos de la base de datos de materiales
    MATERIAL_NAMES_DB_PATH = "assets/reference_data/matName_FullDatabase.npy"
    MATERIAL_LIB_DB_PATH = "assets/reference_data/matLib_FullDatabase.npy"
    
    # Ruta base para archivos de reference_data
    REFERENCE_DATA_BASE_PATH = "assets/reference_data"
    
    # Lista de archivos de gases atmosféricos disponibles
    ATMOSPHERIC_GAS_FILES = [
        "air.txt",      # Aire
        "CH4.txt",      # Metano
        "CO2.txt",      # Dióxido de carbono
        "H2O.txt",      # Vapor de agua
        "O3.txt",        # Ozono
        "enclosure.txt"  # mezcla de varios
    ]


    
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

        # Manejar tanto un solo shape (dict) como múltiples shapes (list)
        if isinstance(shapes, dict):
            # Un solo shape
            print(json.dumps(shapes, indent=2, ensure_ascii=False))
            file = shapes["string"]["@value"]
            file_name = Path(file).stem
            sug = SuggestDTO(id=file, suggest=file_name)
            list_suggest.append(sug)
        elif isinstance(shapes, list):
            # Múltiples shapes
            for shape in shapes:
                print(json.dumps(shape, indent=2, ensure_ascii=False))
                file = shape["string"]["@value"]
                file_name = Path(file).stem
                sug = SuggestDTO(id=file, suggest=file_name)
                list_suggest.append(sug)

        return list_suggest

    def get_material_signature(self, material_name: str) -> np.ndarray:
        """
        Obtiene la firma espectral de un material específico.
        
        Args:
            material_name (str): Nombre del material a buscar (ej: "stone", "concrete", etc.)
            
        Returns:
            np.ndarray: Firma espectral del material
            
        Raises:
            HTTPException: Si el material no se encuentra en la base de datos
        """
        try:
            # Cargar la base de datos
            database_names, database_lib = self.load_material_database()
            
            # Buscar el índice del material
            material_indices = np.where(database_names == material_name)[0]
            
            if len(material_indices) == 0:
                available_materials = database_names[:10]  # Mostrar algunos ejemplos
                raise HTTPException(
                    status_code=404, 
                    detail=f"Material '{material_name}' no encontrado. Materiales disponibles (ejemplos): {list(available_materials)}"
                )
            
            # Obtener la firma espectral
            material_index = material_indices[0]
            material_signature = database_lib[:, material_index]
            
            self.logger.info(f"Firma espectral obtenida para material '{material_name}': shape {material_signature.shape}")
            
            return material_signature
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error obteniendo firma espectral para material '{material_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Error al obtener la firma espectral: {e}")

    def load_material_database(self):
        """
        Carga la base de datos de materiales espectrales.
        
        Returns:
            tuple: (material_names, material_library) donde:
                - material_names: array con los nombres de los materiales
                - material_library: array con las firmas espectrales de los materiales
        """
        try:
            # Cargar nombres de materiales
            database_names = np.load(self.MATERIAL_NAMES_DB_PATH, allow_pickle=True).item()["matName"]
            database_names = database_names.squeeze()
            database_names = np.hstack(database_names)  # lista de nombres de los materiales

            # Cargar librería de materiales
            database_lib = np.load(self.MATERIAL_LIB_DB_PATH, allow_pickle=True).item()["matLib"]
            database_lib = database_lib[::-1, :]  # Reverso el orden de la base de datos

            self.logger.info(f"Base de datos cargada: {database_names.shape} nombres, {database_lib.shape} firmas espectrales")
            
            return database_names, database_lib
            
        except Exception as e:
            self.logger.error(f"Error cargando la base de datos de materiales: {e}")
            raise HTTPException(status_code=500, detail=f"Error al cargar la base de datos de materiales: {e}")

    @staticmethod
    def lista_a_string(valores):
        return ", ".join(str(v) for v in valores)

    @staticmethod
    def blackbody_radiance_nm(wavelengths_nm, temperature):
        """
        Compute spectral radiance B(λ, T) of a black body
        using scipy constants.

        Args:
            wavelengths_nm: array-like of wavelengths in nanometers (nm).
            temperature:    temperature in Kelvin (K).

        Returns:
            numpy array of spectral radiance in W·sr⁻¹·m⁻²·nm⁻¹.
        """
        # Convert wavelengths to meters
        wavelengths_m = np.array(wavelengths_nm, dtype=float) * 1e-9

        # Planck's law for spectral radiance per meter: W·sr⁻¹·m⁻²·m⁻¹
        B_m = (
            (2 * const.h * const.c**2)
            / (wavelengths_m**5)
            / (np.exp(const.h * const.c / (wavelengths_m * const.k * temperature)) - 1)
        )

        # Convert from per meter to per nanometer: 1 m = 1e9 nm
        B_nm = B_m * 1e-9

        return B_nm

    def create_spectral_emitter(self, wavelengths: np.ndarray, emission: np.ndarray, type: str = "area") -> dict:
        """
        Crea un diccionario para un emisor con radiancia espectral irregular.
        
        Args:
            wavelengths (np.ndarray): Array con las longitudes de onda en nanómetros
            emission (np.ndarray): Array con los valores de emisión correspondientes
            
        Returns:
            dict: Diccionario del emisor listo para Mitsuba/XML
            
        Raises:
            HTTPException: Si las longitudes de los arrays no coinciden
        """
        try:
            # Validar que ambos arrays tengan la misma longitud
            if len(wavelengths) != len(emission):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Las longitudes no coinciden: wavelengths={len(wavelengths)}, emission={len(emission)}"
                )
            
            # Crear el diccionario del emisor con estructura para XML
            emitter_dict = {
                "@type": type,
                "spectrum": {
                    "@type": "irregular",
                    "@name": "radiance",
                    "string": [
                        {"@name": "wavelengths", "@value": self.lista_a_string(wavelengths)},
                        {"@name": "values", "@value": self.lista_a_string(emission)},
                    ]
                }
            }
            
            return emitter_dict
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error creando emisor espectral: {e}")
            raise HTTPException(status_code=500, detail=f"Error al crear emisor espectral: {e}")

    def create_homogeneous_medium(self, wavelengths: np.ndarray, sigma_t: np.ndarray, 
                                  medium_id: str = "niebla", g_value: float = 0.95) -> dict:
        """
        Crea un diccionario para un medium homogéneo con coeficiente de extinción espectral.
        
        Args:
            wavelengths (np.ndarray): Array con las longitudes de onda en nanómetros
            sigma_t (np.ndarray): Array con los valores de coeficiente de extinción sigma_t
            medium_id (str): Identificador del medium (por defecto "niebla")
            g_value (float): Parámetro g de la fase Henyey-Greenstein (0.9-0.95 para IR lejano)
            
        Returns:
            dict: Diccionario del medium listo para Mitsuba/XML
            
        Raises:
            HTTPException: Si las longitudes de los arrays no coinciden
        """
        try:
            # Validar que ambos arrays tengan la misma longitud
            if len(wavelengths) != len(sigma_t):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Las longitudes no coinciden: wavelengths={len(wavelengths)}, sigma_t={len(sigma_t)}"
                )
            
            # Crear el diccionario del medium con estructura para XML según documentación Mitsuba
            medium_dict = {
                "@type": "homogeneous",
                "@id": medium_id,
                "rgb": {
                    "@name": "albedo", 
                    "@value": "0.0, 0.0, 0.0"
                },
                "spectrum": {
                    "@type": "irregular",
                    "@name": "sigma_t",
                    "string": [
                        {"@name": "wavelengths", "@value": self.lista_a_string(wavelengths)},
                        {"@name": "values", "@value": self.lista_a_string(sigma_t)},
                    ]
                },
                "phase": {
                    "@type": "hg",
                    "float": {
                        "@name": "g",
                        "@value": str(g_value)
                    }
                }
            }
            
            self.logger.info(f"Medium homogéneo creado con ID '{medium_id}' y g={g_value}")
            
            return medium_dict
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error creando medium homogéneo: {e}")
            raise HTTPException(status_code=500, detail=f"Error al crear medium homogéneo: {e}")

    def get_attenuation_for_wavelengths(self, wavelengths: np.ndarray, attenuation_file: str = "air") -> np.ndarray:
        """
        Obtiene los valores de atenuación (sigma_t) para una lista específica de longitudes de onda.
        Encuentra los valores más cercanos en el archivo de atenuación.
        
        Args:
            wavelengths (np.ndarray): Array con las longitudes de onda deseadas en nanometros
            attenuation_file (str): Nombre del archivo de atenuación (sin extensión)
            
        Returns:
            np.ndarray: Array con los valores de atenuación correspondientes
            
        Raises:
            HTTPException: Si el archivo no existe o hay error en el procesamiento
        """
        # Convertir longitudes de onda de nanómetros a micrómetros
        wavelengths_um = wavelengths / 1000.0

        try:
            # Construir la ruta del archivo
            file_path = f"{self.REFERENCE_DATA_BASE_PATH}/{attenuation_file}.txt"
            
            # Validar que el archivo existe
            if not Path(file_path).exists():
                available_files = [f.replace('.txt', '') for f in self.ATMOSPHERIC_GAS_FILES]
                raise HTTPException(
                    status_code=404,
                    detail=f"Archivo '{attenuation_file}.txt' no encontrado. Disponibles: {available_files}"
                )
            
            # Cargar los datos del archivo
            trans_array = np.loadtxt(file_path)
            
            # Extraer columnas: [wavelength, transmittance, attenuation]
            file_wavelengths = trans_array[:, 0]  # Primera columna: longitudes de onda
            file_attenuation = trans_array[:, 2]  # Tercera columna: atenuación
            
            # Encontrar los índices más cercanos para cada longitud de onda deseada
            attenuation_values = []
            
            for target_wavelength in wavelengths_um:
                # Encontrar el índice del valor más cercano
                closest_index = np.argmin(np.abs(file_wavelengths - target_wavelength))
                attenuation_values.append(file_attenuation[closest_index])
            
            attenuation_array = np.array(attenuation_values)
            
            self.logger.info(f"Atenuación obtenida para {len(wavelengths_um)} longitudes de onda usando '{attenuation_file}.txt'")
            
            return attenuation_array
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error obteniendo atenuación de '{attenuation_file}': {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar archivo de atenuación: {e}")