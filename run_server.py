"""
Script para lanzar el servidor FastAPI de Mitsuba Scene Viewer
"""
import sys
from pathlib import Path

# Agregar el directorio src al path para imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando servidor FastAPI para Mitsuba Scene Viewer...")
    print("📡 API disponible en: http://localhost:8000")
    print("📋 Documentación en: http://localhost:8000/docs")
    print("💡 Para detener el servidor, presiona Ctrl+C")
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
