"""
Decoradores centralizados para validación, logging y error handling.
Reduce repetición de código en servicios y controllers.
"""

import functools
import logging
import os
from typing import Callable
from fastapi import HTTPException


def validate_file_exists(param_name: str = "file_path"):
    """
    Decorador que valida que un parámetro es una ruta de archivo existente.
    
    Uso:
    ```python
    @validate_file_exists('xml_path')
    def process_scene(self, xml_path: str):
        ...
    ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Buscar el parámetro en args o kwargs
            param_value = kwargs.get(param_name)
            if not param_value:
                # Buscar en posición de argumentos
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args):
                        param_value = args[idx]
            
            if param_value and not os.path.exists(param_value):
                raise HTTPException(
                    status_code=404,
                    detail=f"Archivo no encontrado: {param_value}"
                )
            if param_value and not os.path.isfile(param_value):
                raise HTTPException(
                    status_code=400,
                    detail=f"La ruta no es un archivo válido: {param_value}"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_directory_exists(param_name: str = "dir_path"):
    """
    Decorador que valida que un parámetro es una ruta de directorio existente.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            param_value = kwargs.get(param_name)
            if not param_value:
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args):
                        param_value = args[idx]
            
            if param_value and not os.path.exists(param_value):
                raise HTTPException(
                    status_code=404,
                    detail=f"Directorio no encontrado: {param_value}"
                )
            if param_value and not os.path.isdir(param_value):
                raise HTTPException(
                    status_code=400,
                    detail=f"La ruta no es un directorio válido: {param_value}"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_file_extension(param_name: str, allowed_extensions: list):
    """
    Decorador que valida la extensión de un archivo.
    
    Uso:
    ```python
    @validate_file_extension('filename', ['.zip', '.rar'])
    def upload_archive(self, filename: str):
        ...
    ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            param_value = kwargs.get(param_name)
            if not param_value:
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args):
                        param_value = args[idx]
            
            if param_value:
                _, ext = os.path.splitext(param_value)
                if ext.lower() not in [e.lower() for e in allowed_extensions]:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Extensión no permitida: {ext}. "
                            f"Permitidas: {allowed_extensions}"
                        )
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_execution(level: int = logging.INFO):
    """
    Decorador que loguea la ejecución de una función.
    
    Uso:
    ```python
    @log_execution(logging.DEBUG)
    def complex_operation(self, param1, param2):
        ...
    ```
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger.log(
                level,
                f"Iniciando: {func_name} con args={args[1:]} kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"✓ Completado: {func_name}")
                return result
            except Exception as e:
                logger.error(f"✗ Error en {func_name}: {e}")
                raise
        return wrapper
    return decorator


def handle_file_errors():
    """
    Decorador que maneja errores de archivo comúnes y los convierte a HTTPException.
    
    Uso:
    ```python
    @handle_file_errors()
    def list_files(self, directory):
        ...
    ```
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"Archivo no encontrado: {e}")
                raise HTTPException(
                    status_code=404, detail=f"Archivo no encontrado: {e}"
                )
            except IsADirectoryError as e:
                logger.error(f"Es un directorio, no un archivo: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Es un directorio, no un archivo: {e}"
                )
            except PermissionError as e:
                logger.error(f"Permiso denegado: {e}")
                raise HTTPException(
                    status_code=403, detail=f"Permiso denegado: {e}"
                )
            except OSError as e:
                logger.error(f"Error de sistema de archivos: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error de sistema de archivos: {e}"
                )
        return wrapper
    return decorator


def require_scene_loaded():
    """
    Decorador que valida que una escena está cargada (archivo existe).
    
    Uso:
    ```python
    @require_scene_loaded()
    def render_scene(self):
        ...
    ```
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from src.config import PathManager
            path_manager = PathManager
            
            scene_path = path_manager.get_scene_path()
            if not os.path.exists(scene_path):
                logger.error(f"Escena no cargada: {scene_path}")
                raise HTTPException(
                    status_code=400,
                    detail="No hay escena cargada. Carga una escena primero."
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
