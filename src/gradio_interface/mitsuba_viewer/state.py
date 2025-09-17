"""Estado compartido para el visor Mitsuba."""
from typing import Dict, List

class ViewerState:
    def __init__(self):
        self.object_id_mapping: Dict[str, str] = {}
        # Cache: object_id -> ruta local del archivo descargado
        self.object_file_cache: Dict[str, str] = {}
        # Grupos (retrocompat): nombre_grupo -> lista de object_ids
        self.object_groups: Dict[str, list[str]] = {}
        # Grupos por instancia y familia
        self.object_groups_instance: Dict[str, List[str]] = {}
        self.object_groups_family: Dict[str, List[str]] = {}
        self.selected_group_level: str = "instance"  # "instance" | "family"

viewer_state = ViewerState()
