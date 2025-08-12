"""
Object Controller - Placeholder
Controller para manejo de objetos 3D
"""

from fastapi import APIRouter

obj_router = APIRouter(
    prefix="/obj",
    tags=["objects"]
)