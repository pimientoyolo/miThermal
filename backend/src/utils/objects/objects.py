import logging
import json
import numpy as np
import hashlib
from pathlib import Path
from fastapi import HTTPException
from src.utils.scene.parser import SceneParser
from src.api.dto.suggestDTO import SuggestDTO
from scipy import constants as const
import shutil
from src.config import (
    PathManager,
    OUTPUT_STATIC_DIR,
    OUTPUT_SPD_DIR,
    get_config_scene_dict
)

try:
    import open3d
except Exception:
    open3d = None

logger = logging.getLogger(__name__)


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
    
    # Cache de base de datos de materiales (lazy loading)
    _material_database_cache = None


    
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
            
            if open3d is None:
                raise HTTPException(status_code=500, detail="Dependencia open3d no disponible")
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
            
            if open3d is None:
                raise HTTPException(status_code=500, detail="Dependencia open3d no disponible")
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

        logging.debug(f"Validando existencia del archivo: {path}")

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
        Carga la base de datos de materiales espectrales con lazy loading.
        
        La primera vez que se llama, carga la base de datos desde disco y la cachea en memoria.
        Llamadas subsecuentes retornan el cache, evitando lecturas repetidas (~200-500ms ahorro).
        
        Returns:
            tuple: (material_names, material_library) donde:
                - material_names: array con los nombres de los materiales
                - material_library: array con las firmas espectrales de los materiales
        """
        # Retornar cache si ya está cargado
        if ObjectUtils._material_database_cache is not None:
            return ObjectUtils._material_database_cache
        
        try:
            self.logger.info("Cargando base de datos de materiales (primera vez)...")
            
            # Cargar nombres de materiales
            database_names = np.load(self.MATERIAL_NAMES_DB_PATH, allow_pickle=True).item()["matName"]
            database_names = database_names.squeeze()
            database_names = np.hstack(database_names)  # lista de nombres de los materiales

            # Cargar librería de materiales
            database_lib = np.load(self.MATERIAL_LIB_DB_PATH, allow_pickle=True).item()["matLib"]

            self.logger.info(f"Base de datos cargada: {database_names.shape} nombres, {database_lib.shape} firmas espectrales")

            # Guardar en cache para próximas llamadas
            ObjectUtils._material_database_cache = (database_names, database_lib)
            
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
        Calcula la radiancia espectral B(λ, T) de un cuerpo negro usando la ley de Planck.
        
        Args:
            wavelengths_nm: Longitudes de onda en nanómetros (nm).
            temperature: Temperatura en Kelvin (K).
            
        Returns:
            Radiancia espectral en W/(m²·sr·m).
        """

        logger.info(f"Calculando radiancia espectral para λ={np.min(wavelengths_nm)}-{np.max(wavelengths_nm)}, T={temperature}")
        # Convertir longitudes de onda a metros
        wavelengths_m = np.array(wavelengths_nm, dtype=float) * 1e-9

        # Ley de Planck: B_λ(T) = (2hc²) / (λ⁵ * (exp(hc/λkT) - 1))
        # Usamos constantes de alta precisión de scipy
        c1 = 2 * const.h * const.c**2
        c2 = (const.h * const.c) / const.k
        
        exponent = c2 / (wavelengths_m * temperature)
        # B_lambda en W/(m^2 * sr * m)
        radiance_m = c1 / (wavelengths_m**5 * (np.exp(exponent) - 1))
        
        return radiance_m
    
    def _save_spectrum_spd(self, wavelengths_nm: np.ndarray, values: np.ndarray, filename: str, spectrum_type: str = "spectral distribution") -> tuple[str, str]:
        """
        Guarda un espectro en un archivo .spd externo con un nombre significativo.
        Retorna (shasum, safe_filename_with_ext)
        
        Args:
            wavelengths_nm: Array de longitudes de onda en nanómetros.
            values: Array de valores (radiancia o reflectancia 0-1).
            filename: Nombre base del archivo (sin extensión).
            spectrum_type: Tipo de espectro para el encabezado (reflectance/radiance).
            
        Returns:
            tuple: (Hash SHA-256, nombre de archivo final con extensión)
        """
        # Mitsuba requiere que las longitudes de onda estén en orden ASCENDENTE
        # Ordenar ambos arrays basados en las longitudes de onda
        idx = np.argsort(wavelengths_nm)
        w_sorted = wavelengths_nm[idx]
        v_sorted = values[idx]

        # Generar contenido del archivo con tipo específico en el comentario
        lines = [f"# This file contains a measured {spectrum_type}"]
        for wl, val in zip(w_sorted, v_sorted):
            lines.append(f"{wl:.4f} {val:.6f}")
        
        content = "\n".join(lines)
        sha256_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Sanitizar nombre de archivo (reemplazar caracteres no permitidos)
        safe_name = filename.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_filename = f"{safe_name}.spd"
        file_path = OUTPUT_SPD_DIR / safe_filename
        
        # Determinar si necesitamos escribir el archivo
        should_write = True
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
            existing_hash = hashlib.sha256(existing_content.encode()).hexdigest()
            if existing_hash == sha256_hash:
                should_write = False
        
        if should_write:
            # Asegurar que el directorio existe
            OUTPUT_SPD_DIR.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Espectro ({spectrum_type}) guardado/actualizado en: {file_path}")
        
        return sha256_hash, safe_filename

    def create_reflectance_material(self, wavelengths: np.ndarray, reflectance: np.ndarray, identifier: str = "default") -> tuple[dict, str]:
        """
        Crea un material con reflectancia usando un archivo .spd con nombre significativo.
        Retorna (mitsuba_dict, shasum).
        """
        try:
            if len(wavelengths) != len(reflectance):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Las longitudes no coinciden: wavelengths={len(wavelengths)}, reflectance={len(reflectance)}"
                )
            
            # Guardar en archivo .spd con nombre descriptivo y tipo
            spd_base_name = f"object_{identifier}"
            shasum, safe_filename = self._save_spectrum_spd(wavelengths, reflectance, spd_base_name, "spectrum reflectance")
            
            # Ruta relativa para el XML: spds/nombre_archivo.spd
            # Mitsuba busca relativo al archivo .xml de la escena
            relative_spd_path = f"spds/{safe_filename}"

            material_dict = {
                "@type": "diffuse",
                "spectrum": {
                    "@name": "reflectance",
                    "@filename": relative_spd_path
                }
            }
            
            return material_dict, shasum
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error creando material de reflectancia: {e}")
            raise HTTPException(status_code=500, detail=f"Error al crear material de reflectancia: {e}")

    def create_spectral_emitter(self, wavelengths: np.ndarray, emission: np.ndarray, identifier: str = "default", type: str = "area") -> tuple[dict, str]:
        """
        Crea un emisor espectral usando un archivo .spd con nombre significativo.
        Retorna (mitsuba_dict, shasum).
        """
        try:
            if len(wavelengths) != len(emission):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Las longitudes no coinciden: wavelengths={len(wavelengths)}, emission={len(emission)}"
                )
            
            # Guardar en archivo .spd con nombre descriptivo y tipo
            prefix = "atmosphere" if type == "constant" else "emitter"
            spd_base_name = f"{prefix}_{identifier}"
            shasum, safe_filename = self._save_spectrum_spd(wavelengths, emission, spd_base_name, "spectrum radiance")
            
            # Ruta relativa para el XML
            relative_spd_path = f"spds/{safe_filename}"

            emitter_dict = {
                "@type": type,
                "spectrum": {
                    "@name": "radiance",
                    "@filename": relative_spd_path
                }
            }
            
            return emitter_dict, shasum
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error creando emisor espectral: {e}")
            raise HTTPException(status_code=500, detail=f"Error al crear emisor espectral: {e}")

    def save_new_emissivity(self, object_id: str, wavelengths_nm: np.ndarray, emissivity: np.ndarray) -> None:
        """
        Guarda un nuevo archivo de emisividad para un objeto específico.
        El archivo se guarda en OUTPUT_STATIC_DIR con la misma estructura que PathManager.get_default_emissivity_path().

        Args:
            object_id (str): Identificador del objeto (ruta relativa como en la escena)
            wavelengths_nm (np.ndarray): Array con las longitudes de onda en nanómetros
            emissivity (np.ndarray): Array con los valores de emisividad correspondientes

        Raises:
            HTTPException: Si ocurre algún error durante el guardado
        """
        try:
            path_static = OUTPUT_STATIC_DIR
            base_out = Path(path_static)

            obj_path = Path(object_id)
            # Si es absoluta (p.ej., Windows con drive), convertir a relativa respecto a la raíz
            if obj_path.is_absolute():
                obj_path = obj_path.relative_to(obj_path.anchor)

            dst_path = base_out / obj_path.with_suffix(".txt")
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Guardar a archivo tab-delimitado (wavelength_nm, emissivity)
            data = np.column_stack((wavelengths_nm, emissivity))
            np.savetxt(dst_path, data, delimiter='\t')

            self.logger.info(f"Archivo de emisividad guardado en '{dst_path}' para objeto '{object_id}'")
        except Exception as e:
            self.logger.error(f"Error guardando archivo de emisividad para objeto '{object_id}': {e}")
            raise HTTPException(status_code=500, detail=f"Error al guardar archivo de emisividad: {e}")
    
    def save_default_emissivity(self, object_id: str) -> None:
        
        path_default = PathManager.get_default_emissivity_path()
        self.valid_exist_file(path_default)
        path_static = OUTPUT_STATIC_DIR

        try:
            src_path = Path(path_default)
            base_out = Path(path_static)

            obj_path = Path(object_id)
            # Si es absoluta (p.ej., Windows con drive), convertir a relativa respecto a la raíz
            if obj_path.is_absolute():
                obj_path = obj_path.relative_to(obj_path.anchor)

            dst_path = base_out / obj_path.with_suffix(".txt")
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dst_path)
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error copiando archivo de emisividad por defecto: {e}")
            raise HTTPException(status_code=500, detail="Error al copiar archivo de emisividad por defecto")

    def read_reflectance_file_as_emissivity(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Lee un archivo de dos columnas [wavelength, reflectance] y retorna:
        - wavelengths_nm: longitudes de onda en nanómetros (nm)
        - emissivity: emisividad (1 - reflectancia), en rango [0, 1]
        """
        # Validar que el archivo exista
        self.valid_exist_file(file_path)

        try:
            data = np.loadtxt(file_path)

            if data.ndim == 1:
                data = data.reshape(-1, 2)

            wavelengths = data[:, 0].astype(float)
            reflectance_vals = data[:, 1].astype(float)

            # 1. Manejo de unidades de longitud de onda
            # Si el máximo es < 100, asumimos µm y convertimos a nm
            if np.max(wavelengths) < 100:
                wavelengths = wavelengths * 1000.0
            
            # 2. Manejo de unidades de reflectancia
            # Si el máximo es > 1.0, asumimos que está en porcentaje (0-100) y normalizamos a (0-1)
            if np.max(reflectance_vals) > 1.0:
                reflectance_vals = reflectance_vals / 100.0
            
            # 3. Convertir a Emisividad (Kirchhoff's Law: E = 1 - R para cuerpos opacos)
            emissivity = 1.0 - reflectance_vals
            emissivity = np.clip(emissivity, 0.0, 1.0)

            # Ordenar por longitud de onda
            order = np.argsort(wavelengths)
            return wavelengths[order], emissivity[order]

        except Exception as e:
            self.logger.error(f"Error procesando archivo espectral '{file_path}': {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar archivo espectral: {e}")

    def read_object_emissivity_file(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Conveniencia: Dado un object_id (ruta relativa como en la escena),
        abre el archivo .txt correspondiente en OUTPUT_STATIC_DIR y retorna
        (wavelengths_nm, emissivity) usando read_reflectance_file_as_emissivity.
        """
        config_scene = get_config_scene_dict()
        emissivity_file = config_scene.get("objects", {}).get(object_id, {}).get("emissivity_file", None)

        # Si la configuración tiene un archivo asociado y existe, úsalo
        if emissivity_file and Path(emissivity_file).exists():
            return self.read_reflectance_file_as_emissivity(emissivity_file)

        # Intentar localizar un archivo .txt bajo OUTPUT_STATIC_DIR siguiendo el object_id
        base_out = Path(OUTPUT_STATIC_DIR)
        obj_path = Path(object_id)
        if obj_path.is_absolute():
            obj_path = obj_path.relative_to(obj_path.anchor)
        candidate = base_out / obj_path.with_suffix('.txt')
        if candidate.exists():
            return self.read_reflectance_file_as_emissivity(str(candidate))

        # Intentar crear un archivo de emisividad por defecto para el objeto
        try:
            self.save_default_emissivity(object_id)
            if candidate.exists():
                return self.read_reflectance_file_as_emissivity(str(candidate))
        except Exception:
            # fallthrough al error final
            pass

        self.logger.warning(f"No se encontró archivo de emisividad para objeto '{object_id}'")
        raise HTTPException(status_code=404, detail="Archivo de emisividad no encontrado")
