"""
Scene Helpers - Enums y utilities para manejo centralizado de tipos y preparación de escenas
"""

from enum import Enum
from typing import List, Optional
import logging
from abc import ABC, abstractmethod


class SceneType(str, Enum):
    """
    Tipos de escenas disponibles con sus características.
    Reemplaza strings literales dispersos en el código.
    """
    RGB = "rgb"
    DEPTH = "depth"
    THERMAL = "thermal"
    BLACKBODY_AIR = "blackbody_air"
    TRANSMITTANCE_BLACKBODY_AIR = "transmittance_blackbody_air"
    TEMPERATURE_MAP = "temperature_map"
    
    @property
    def description(self) -> str:
        """Descripción legible del tipo de escena"""
        descriptions = {
            "rgb": "Renderizado RGB básico",
            "depth": "Mapa de profundidad",
            "thermal": "Imagen térmica",
            "blackbody_air": "Blackbody Air",
            "transmittance_blackbody_air": "Transmittance Blackbody Air",
            "temperature_map": "Mapa de temperatura"
        }
        return descriptions.get(self.value, self.value)


class ScenePreparationTask(ABC):
    """
    Abstract base class para tareas de preparación de escena.
    Permite nueva síntesis de preparación de escenas.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self) -> None:
        """
        Ejecuta la tarea de preparación.
        Debe ser implementado por subclases.
        """
        pass
    
    @abstractmethod
    def get_scene_type(self) -> SceneType:
        """Retorna el tipo de escena que prepara"""
        pass
    
    @property
    def scene_type_name(self) -> str:
        """Nombre legible de la escena"""
        return self.get_scene_type().description


class ScenePreparationPipeline:
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.tasks: List[ScenePreparationTask] = []
        self.executed_tasks: List[str] = []
        self.failed_tasks: List[tuple[str, Exception]] = []
    
    def add_task(self, task: ScenePreparationTask) -> "ScenePreparationPipeline":
        """
        Agrega una tarea al pipeline.
        
        Args:
            task: Tarea a agregar
            
        Returns:
            Self para encadenamiento (fluent API)
        """
        self.tasks.append(task)
        return self
    
    def add_tasks(self, *tasks: ScenePreparationTask) -> "ScenePreparationPipeline":
        """
        Agrega múltiples tareas al pipeline.
        
        Args:
            *tasks: Variables tareas a agregar
            
        Returns:
            Self para encadenamiento
        """
        for task in tasks:
            self.add_task(task)
        return self
    
    def execute_all(self, stop_on_error: bool = False) -> bool:
        """
        Ejecuta todas las tareas en orden.
        
        Args:
            stop_on_error: Si True, detiene en primer error. Si False, continúa.
            
        Returns:
            True si todas las tareas completaron éxito, False si hay errores
        """
        self.executed_tasks = []
        self.failed_tasks = []
        
        for task in self.tasks:
            task_name = task.scene_type_name
            try:
                self.logger.info(f"Ejecutando preparación de escena: {task_name}")
                task.execute()
                self.executed_tasks.append(task_name)
                self.logger.info(f"✓ {task_name} completado")
            except Exception as e:
                self.logger.error(f"✗ Error en {task_name}: {e}")
                self.failed_tasks.append((task_name, e))
                if stop_on_error:
                    raise
        
        return len(self.failed_tasks) == 0
    
    def get_status(self) -> dict:
        """
        Retorna estado del pipeline después de ejecutar.
        
        Returns:
            Dict con total, completados, fallidos y errores
        """
        return {
            "total_tasks": len(self.tasks),
            "executed": len(self.executed_tasks),
            "failed": len(self.failed_tasks),
            "executed_tasks": self.executed_tasks,
            "failed_tasks": [(name, str(err)) for name, err in self.failed_tasks]
        }
    
    def reset(self) -> None:
        """Reinicia el pipeline (pero no elimina tareas)"""
        self.executed_tasks = []
        self.failed_tasks = []


class RenderTask(ABC):
    """
    Abstract base class para tareas de renderizado.
    Aplicar factory pattern para renderizadores.
    """
    
    def __init__(self, scene_type: SceneType, logger: Optional[logging.Logger] = None):
        self.scene_type = scene_type
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def validate(self) -> None:
        """Valida que la escena existe y es válida"""
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Ejecuta el renderizado"""
        pass
    
    def execute(self) -> None:
        """Plantilla method: validate + render"""
        self.validate()
        self.render()


class ValidationMixin:
    """
    Mixin para validaciones comunes de archivos y rutas.
    Reduce duplicación en múltiples clases de servicio.
    """
    
    def validate_file_path(self, path: str, must_exist: bool = True) -> None:
        """Valida ruta de archivo"""
        import os
        if must_exist and not os.path.exists(path):
            raise ValueError(f"Archivo no existe: {path}")
        if os.path.exists(path) and not os.path.isfile(path):
            raise ValueError(f"No es un archivo: {path}")
    
    def validate_directory_path(self, path: str, must_exist: bool = True) -> None:
        """Valida ruta de directorio"""
        import os
        if must_exist and not os.path.exists(path):
            raise ValueError(f"Directorio no existe: {path}")
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError(f"No es un directorio: {path}")
    
    def validate_file_extension(self, path: str, allowed_extensions: List[str]) -> None:
        """
        Valida que archivo tiene extensión permitida.
        
        Args:
            path: Ruta del archivo
            allowed_extensions: Lista de extensiones permitidas (ej: ['.txt', '.csv'])
        """
        import os
        _, ext = os.path.splitext(path)
        if ext.lower() not in [e.lower() for e in allowed_extensions]:
            raise ValueError(f"Extensión no permitida: {ext}. Permitidas: {allowed_extensions}")
