"""
Render Controller - Placeholder
Controller para renderizado de escenas
"""

from fastapi import APIRouter

render_router = APIRouter(
    prefix="/render",
    tags=["rendering"]
)