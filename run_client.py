#!/usr/bin/env python3
"""
Script para lanzar el cliente Gradio de Mitsuba Scene Viewer
"""
import sys
from pathlib import Path

# Agregar el directorio src al path para imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

if __name__ == "__main__":
    from src.api.client import create_mitsuba_viewer_interface
    
    print("🎨 Iniciando cliente Gradio para Mitsuba Scene Viewer...")
    print("🌐 Interfaz web disponible en: http://localhost:7860")
    print("⚠️  Asegúrate de que el servidor esté ejecutándose en http://localhost:8000")
    print("💡 Para detener el cliente, presiona Ctrl+C")
    
    # Crear y lanzar la interfaz
    interface = create_mitsuba_viewer_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
