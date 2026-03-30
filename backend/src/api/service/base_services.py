"""
Base Services - Clases base reutilizables para servicios
Reduce boilerplate mediante herencia y composición
"""

import logging
import io
from abc import ABC
from typing import Any, Optional, Callable
from fastapi import HTTPException
import os
import shutil
import zipfile
from pathlib import Path


class BaseService(ABC):
    """
    Clase base para servicios con manejo común de errores, logging y validaciones.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _safe_execute(self, func: Callable, error_message: str = "Error during operation", 
                     status_code: int = 500) -> Optional[Any]:
        """
        Ejecuta una función con manejo centralizado de errores.
        
        Args:
            func: Función a ejecutar
            error_message: Mensaje de error personalizado
            status_code: Código HTTP del error
            
        Returns:
            Resultado de la función o None si hay error
        """
        try:
            return func()
        except HTTPException:
            raise
        except Exception as e:
            # Loguear solo el error específico para reducir ruido como pidió el usuario
            self.logger.error(f"{e}")
            raise HTTPException(status_code=status_code, detail=f"{error_message}: {e}")
    
    def _validate_file_exists(self, file_path: str, file_type: str = "file") -> None:
        """
        Valida que un archivo existe.
        
        Args:
            file_path: Ruta del archivo
            file_type: Tipo de archivo para mensajes de error
            
        Raises:
            HTTPException: Si el archivo no existe
        """
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"El {file_type} no existe: {file_path}"
            )
    
    def _validate_is_file(self, file_path: str) -> None:
        """
        Valida que una ruta es un archivo (no directorio).
        
        Args:
            file_path: Ruta a validar
            
        Raises:
            HTTPException: Si no es un archivo válido
        """
        if not os.path.isfile(file_path):
            raise HTTPException(
                status_code=400,
                detail=f"La ruta no es un archivo válido: {file_path}"
            )
    
    def _validate_is_directory(self, dir_path: str) -> None:
        """
        Valida que una ruta es un directorio.
        
        Args:
            dir_path: Ruta a validar
            
        Raises:
            HTTPException: Si no es un directorio válido
        """
        if not os.path.isdir(dir_path):
            raise HTTPException(
                status_code=400,
                detail=f"La ruta no es un directorio válido: {dir_path}"
            )


class ZipHandler:
    """
    Gestión centralizada de operaciones con archivos ZIP.
    Reduce duplicación de código en load, extract y create operaciones.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def clear_and_extract(self, zip_path: str, extract_to: str, 
                         preserve_subdirs: bool = True) -> None:
        """
        Limpia un directorio y extrae un ZIP en él.
        Patrón usado en load_scene() y upload_mi_thermal_scene()
        
        Args:
            zip_path: Ruta del archivo ZIP
            extract_to: Directorio destino (será limpiado)
            preserve_subdirs: Si se preservan subdirectorios al extraer
            
        Raises:
            HTTPException: Si hay errores
        """
        zip_abs_path = os.path.abspath(zip_path)
        extract_abs_path = os.path.abspath(extract_to)

        # Si el ZIP está dentro del destino, leerlo en memoria antes de limpiar
        zip_bytes = None
        zip_inside_extract_dir = (
            zip_abs_path == extract_abs_path
            or zip_abs_path.startswith(extract_abs_path + os.sep)
        )

        if zip_inside_extract_dir:
            try:
                with open(zip_abs_path, "rb") as source_zip:
                    zip_bytes = source_zip.read()
            except OSError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"No se pudo leer ZIP temporal antes de limpiar: {e}",
                )

        # Limpiar directorio destino
        if os.path.exists(extract_to):
            try:
                shutil.rmtree(extract_to)
            except OSError as e:
                self.logger.warning(f"Error limpiando {extract_to}: {e}")
            finally:
                os.makedirs(extract_to, exist_ok=True)
        else:
            os.makedirs(extract_to, exist_ok=True)
        
        # Extraer ZIP
        try:
            zip_source = zip_path if zip_bytes is None else io.BytesIO(zip_bytes)
            with zipfile.ZipFile(zip_source, 'r') as zip_ref:
                if preserve_subdirs:
                    zip_ref.extractall(extract_to)
                else:
                    # Extraer archivos ignorando estructura de carpetas
                    for member in zip_ref.namelist():
                        if not member.endswith('/'):
                            filename = os.path.basename(member)
                            data = zip_ref.read(member)
                            with open(os.path.join(extract_to, filename), 'wb') as f:
                                f.write(data)
        except zipfile.BadZipFile as e:
            raise HTTPException(status_code=400, detail=f"Archivo ZIP inválido: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extrayendo ZIP: {e}")
    
    def create_zip_from_directory(self, source_dir: str, zip_path: str, 
                                 compression: int = zipfile.ZIP_STORED) -> None:
        """
        Crea un ZIP desde un directorio.
        Patrón usado en get_scene_mi_thermal()
        
        Args:
            source_dir: Directorio origen
            zip_path: Ruta del ZIP a crear
            compression: Tipo de compresión (ZIP_STORED, ZIP_DEFLATED, etc.)
        """
        try:
            # Remover ZIP si existe
            if os.path.exists(zip_path):
                os.remove(zip_path)
            
            with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, source_dir)
                        zf.write(abs_path, rel_path)
        except Exception as e:
            self.logger.error(f"Error creando ZIP {zip_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error creando archivo ZIP: {e}")
    
    def extract_and_remove(self, zip_path: str, extract_to: str) -> None:
        """
        Extrae ZIP y lo borra.
        Patrón usado en set_default_scene()
        
        Args:
            zip_path: Ruta del ZIP
            extract_to: Directorio destino
        """
        self.clear_and_extract(zip_path, extract_to)
        try:
            os.remove(zip_path)
        except OSError as e:
            self.logger.warning(f"No se pudo eliminar ZIP {zip_path}: {e}")


class RenderServiceBase(BaseService):
    """
    Clase base para servicios de renderizado.
    Centraliza validación y manejo de excepciones comunes.
    """
    
    def _render_with_validation(self, scene_type: str, render_func: Callable) -> None:
        """
        Ejecuta renderizado con validación y manejo de errores centralizado.

        Args:
            scene_type: Tipo de escena (rgb, thermal, etc.)
            render_func: Función de renderizado a ejecutar
        """
        def _execute():
            render_func()

        error_msg = f"Error al renderizar la escena {scene_type}"
        self._safe_execute(_execute, error_msg)

class FileHandler:
    """
    Gestión centralizada de operaciones con archivos.
    Reduce duplicación en validaciones de rutas y operaciones de copia.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def copy_file_safe(self, src: str, dst: str, overwrite: bool = True) -> None:
        """
        Copia archivo con validación y manejo de errores.
        
        Args:
            src: Archivo origen
            dst: Archivo destino
            overwrite: Permitir sobrescribir archivo existente
            
        Raises:
            HTTPException: Si hay errores
        """
        try:
            if not overwrite and os.path.exists(dst):
                raise HTTPException(
                    status_code=400,
                    detail=f"El archivo {dst} ya existe"
                )
            
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        except HTTPException:
            raise
        except OSError as e:
            self.logger.error(f"Error copiando {src} a {dst}: {e}")
            raise HTTPException(status_code=500, detail=f"Error copiando archivo: {e}")
    
    def get_relative_path(self, absolute_path: str) -> str:
        """
        Convierte ruta absoluta a relativa, manejando diferencias entre sistemas.
        Patrón usado en obj_service.py
        
        Args:
            absolute_path: Ruta absoluta o relativa
            
        Returns:
            Ruta relativa
        """
        obj_path = Path(absolute_path)
        if obj_path.is_absolute():
            obj_path = obj_path.relative_to(obj_path.anchor)
        return str(obj_path)
    
    def ensure_parent_directory(self, file_path: str) -> None:
        """
        Asegura que el directorio padre existe.
        
        Args:
            file_path: Ruta del archivo
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
