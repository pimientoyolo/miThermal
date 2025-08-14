"""Estado compartido para el visor Mitsuba."""
from typing import Dict

class ViewerState:
    def __init__(self):
        self.object_id_mapping: Dict[str, str] = {}

viewer_state = ViewerState()
