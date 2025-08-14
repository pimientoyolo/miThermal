"""
Funciones para la creación y configuración de escenas en Mitsuba.
"""
import mitsuba as mi

import logging

class Scene:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
