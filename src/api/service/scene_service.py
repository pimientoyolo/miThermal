import logging

from fastapi import UploadFile

from ...mitsuba_core.scene_parser import MitsubaSceneParser

logger = logging.getLogger(__name__)

class SceneService:
    def __init__(self):
        self.logger = logger

    def load_scene(self, file: UploadFile):
        pass
