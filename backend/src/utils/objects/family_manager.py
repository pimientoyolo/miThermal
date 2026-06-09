"""
Gestión de familias de objetos basado en convenciones de nomenclatura.

Este módulo detecta automáticamente familias de objetos usando patrones
comunes de nomenclatura (especialmente Blender), y permite aplicar
propiedades a todos los miembros de una familia.
"""

import re
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ObjectFamily:
    """
    Representa una familia de objetos relacionados.
    
    Atributos:
        base_name: Nombre base sin sufijo numérico (ej: "tree_leaf")
        members: Lista de object_ids que pertenecen a la familia
        shared_properties: Propiedades compartidas por todos los miembros
    """
    base_name: str
    members: List[str] = field(default_factory=list)
    shared_properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ordena los miembros alfabéticamente."""
        self.members = sorted(self.members)
    
    @property
    def count(self) -> int:
        """Retorna el número de miembros en la familia."""
        return len(self.members)


class FamilyManager:
    """
    Gestiona familias de objetos con convención de nomenclatura.
    
    Detecta automáticamente familias usando patrones comunes:
    - Blender: object.001, object.002
    - General: object_001, object_002
    - Copy: object_copy_1, object_copy_2
    """
    
    # Patrones de nomenclatura soportados (orden de prioridad)
    PATTERNS = [
        r"^(.+)\.(\d+)$",            # Blender/General: tree_leaf.001, tree_leaf.1
        r"^(.+)_(\d+)$",            # General: wall_north_001, wall_north_1
        r"^(.+)_copy_?(\d*)$",       # Copy: object_copy_1
        r"^(.+)_instance_?(\d+)$",   # Instance: object_instance_1
    ]
    
    def __init__(self, config_scene: Dict[str, Any]):
        """
        Inicializa el gestor de familias.
        
        Args:
            config_scene: Diccionario con configuración de la escena
        """
        self.config = config_scene
        self.families: Dict[str, ObjectFamily] = {}
        self._build_families()
    
    def _build_families(self):
        """Construye el índice de familias desde config_scene."""
        objects = self.config.get("objects", {})
        
        if not objects:
            logger.warning("No hay objetos en config_scene")
            return
        
        # Agrupar objetos por base_name
        groups: Dict[str, List[str]] = {}
        
        for obj_id in objects.keys():
            base_name = self._extract_base_name(obj_id)
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(obj_id)
        
        # Crear familias solo si hay >1 miembro
        for base_name, members in groups.items():
            if len(members) > 1:
                self.families[base_name] = ObjectFamily(
                    base_name=base_name,
                    members=members,
                    shared_properties=self._find_shared_properties(members)
                )
                logger.debug(
                    f"Familia detectada: '{base_name}' "
                    f"con {len(members)} miembros"
                )
    
    def _extract_base_name(self, object_id: str) -> str:
        """
        Extrae nombre base según patrones de nomenclatura.
        
        Args:
            object_id: ID del objeto (ej: "tree_leaf.001")
            
        Returns:
            Nombre base sin sufijo (ej: "tree_leaf")
        """
        name, ext = os.path.splitext(object_id)
        for pattern in self.PATTERNS:
            match = re.match(pattern, name)
            if match:
                return match.group(1)  # Retorna grupo sin sufijo
        
        # Si no coincide con ningún patrón, es su propia familia
        return name
    
    def _find_shared_properties(self, members: List[str]) -> Dict[str, Any]:
        """
        Encuentra propiedades compartidas por todos los miembros.
        
        Args:
            members: Lista de object_ids de la familia
            
        Returns:
            Dict con propiedades que tienen el mismo valor en todos
        """
        if not members:
            return {}
        
        objects = self.config.get("objects", {})
        first_obj = objects.get(members[0], {})
        shared = {}
        
        # Propiedades a comparar
        comparable_props = [
            "temperature",
            "emissivity_file",
            "reflectance_file",
            "emissivity_hash",
            "reflectance_hash"
        ]
        
        for key in comparable_props:
            if key not in first_obj:
                continue
            
            value = first_obj[key]
            # Verificar si todos los miembros tienen el mismo valor
            if all(
                objects.get(m, {}).get(key) == value
                for m in members[1:]
            ):
                shared[key] = value
        
        return shared
    
    def get_family(self, object_id: str) -> Optional[ObjectFamily]:
        """
        Retorna la familia de un objeto.
        
        Args:
            object_id: ID del objeto
            
        Returns:
            ObjectFamily si pertenece a una familia, None si es único
        """
        base_name = self._extract_base_name(object_id)
        return self.families.get(base_name)
    
    def list_all_families(self) -> List[ObjectFamily]:
        """
        Lista todas las familias detectadas.
        
        Returns:
            Lista de ObjectFamily ordenadas por base_name
        """
        return sorted(
            self.families.values(),
            key=lambda f: f.base_name
        )
    
    def update_family(
        self,
        base_name: str,
        properties: Dict[str, Any]
    ) -> List[str]:
        """
        Actualiza todos los miembros de una familia.
        
        Args:
            base_name: Nombre base de la familia
            properties: Dict con propiedades a actualizar
            
        Returns:
            Lista de object_ids actualizados
            
        Raises:
            ValueError: Si la familia no existe
        """
        family = self.families.get(base_name)
        if not family:
            raise ValueError(f"Familia '{base_name}' no encontrada")
        
        objects = self.config.get("objects", {})
        updated = []
        
        for obj_id in family.members:
            if obj_id in objects:
                objects[obj_id].update(properties)
                updated.append(obj_id)
        
        logger.info(
            f"Familia '{base_name}' actualizada: {len(updated)} objetos"
        )
        return updated
    
    def get_family_summary(self) -> Dict[str, int]:
        """
        Resumen de familias para mostrar en UI.
        
        Returns:
            Dict con base_name -> número de miembros
        """
        return {
            family.base_name: family.count
            for family in self.families.values()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Estadísticas generales sobre familias.
        
        Returns:
            Dict con estadísticas
        """
        if not self.families:
            return {
                "total_families": 0,
                "total_objects_in_families": 0,
                "avg_family_size": 0,
                "largest_family": None
            }
        
        family_sizes = [f.count for f in self.families.values()]
        largest = max(self.families.values(), key=lambda f: f.count)
        
        return {
            "total_families": len(self.families),
            "total_objects_in_families": sum(family_sizes),
            "avg_family_size": sum(family_sizes) / len(family_sizes),
            "largest_family": {
                "base_name": largest.base_name,
                "count": largest.count
            }
        }
    
    def preview_family_update(
        self,
        object_id: str
    ) -> Dict[str, Any]:
        """
        Previsualiza qué objetos se actualizarían en modo familia.
        
        Args:
            object_id: ID del objeto seleccionado
            
        Returns:
            Dict con información de la familia
        """
        family = self.get_family(object_id)
        
        if not family:
            return {
                "has_family": False,
                "object_id": object_id,
                "message": "Este objeto no pertenece a ninguna familia"
            }
        
        return {
            "has_family": True,
            "family_name": family.base_name,
            "members": family.members,
            "count": family.count,
            "shared_properties": family.shared_properties,
            "message": (
                f"Se actualizarán {family.count} objetos "
                f"de la familia '{family.base_name}'"
            )
        }
