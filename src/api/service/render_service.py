import logging

logger = logging.getLogger(__name__)

class RenderService:
    def __init__(self):
        self.logger = logger

    def render_basic_scene(self) -> str:
        return r"C:\Users\mgefr\Documents\1UIS\proyecto-grado\output\static\prev.png"

