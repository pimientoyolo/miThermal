#!/usr/bin/env python3
"""
Script para lanzar el cliente Gradio de Mitsuba Scene Viewer
"""
import sys
from pathlib import Path
import argparse
import os

# Agregar el directorio src al path para imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _build_base_url(host: str | None, port: int | None, base: str | None) -> str:
    if base:
        return base.rstrip("/")
    host = host or "localhost"
    port = int(port) if port else 8000
    return f"http://{host}:{port}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente Gradio para Mitsuba Scene Viewer")
    parser.add_argument("--backend-host", help="Host del backend (ej: 127.0.0.1)")
    parser.add_argument("--backend-port", type=int, help="Puerto del backend (ej: 8000)")
    parser.add_argument("--backend-base", help="URL base completa del backend (ej: http://1.2.3.4:8000)")
    args = parser.parse_args()

    # Construir base_url y exportarla para que el cliente la use al importarse
    base_url = _build_base_url(args.backend_host, args.backend_port, args.backend_base)
    os.environ.setdefault("MITSUBA_API_BASE", base_url)

    # Importar la interfaz después de fijar la variable de entorno
    from src.gradio_interface.mitsuba_viewer.callbacks import create_mitsuba_viewer_interface

    print("🎨 Iniciando cliente Gradio para Mitsuba Scene Viewer...")
    print("🌐 Interfaz web: http://localhost:7860")
    print(f"⚠️ Servidor API esperado en {base_url}")
    print("💡 Ctrl+C para detener")

    interface = create_mitsuba_viewer_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
