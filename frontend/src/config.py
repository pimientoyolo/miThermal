"""
Compat layer para `config` en el frontend.

Este módulo intenta reutilizar la configuración definida en
`backend/src/config.py`. Inserta la raíz del proyecto en `sys.path`
para permitir la importación del paquete `backend` cuando el frontend
se ejecuta desde `frontend/main.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Añadir la raíz del proyecto al path para que `backend` sea importable
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from backend.src.config import get_config as _get_config
except Exception:
    # Fallback: intentar importar como paquete relativo si la estructura cambia
    try:
        from src.config import get_config as _get_config
    except Exception:
        def get_config() -> Dict[str, Any]:
            # Mínimo fallback para evitar que el frontend falle completamente.
            return {
                "gradio": {
                    "title": "MiThermal (fallback)",
                    "description": "Configuración por defecto (fallback)",
                }
            }
    else:
        def get_config() -> Dict[str, Any]:
            return _get_config()
else:
    def get_config() -> Dict[str, Any]:
        return _get_config()
