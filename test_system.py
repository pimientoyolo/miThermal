#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento del sistema cliente-servidor
"""
import sys
import requests
from pathlib import Path

# Agregar el directorio src al path para imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_server_connection():
    """Prueba la conexión al servidor"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor FastAPI funcionando correctamente")
            return True
        else:
            print(f"❌ Servidor respondió con código: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servidor. ¿Está ejecutándose?")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_api_endpoints():
    """Prueba los endpoints básicos de la API"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        print("✅ Endpoint /health funciona")
    except Exception as e:
        print(f"❌ Error en /health: {e}")
        return False
    
    # Test objects endpoint (sin escena cargada)
    try:
        response = requests.get(f"{base_url}/objects")
        print(f"✅ Endpoint /objects responde (código: {response.status_code})")
    except Exception as e:
        print(f"❌ Error en /objects: {e}")
        return False
    
    # Test scene-info endpoint (sin escena cargada)
    try:
        response = requests.get(f"{base_url}/scene-info")
        print(f"✅ Endpoint /scene-info responde (código: {response.status_code})")
    except Exception as e:
        print(f"❌ Error en /scene-info: {e}")
        return False
    
    return True

def test_imports():
    """Prueba que todos los imports funcionan correctamente"""
    try:
        # Importar para probar que funcionan
        _ = __import__("src.mitsuba_core.scene_parser").MitsubaSceneParser
        print("✅ Parser de escenas importado correctamente")
        
        _ = __import__("src.api.server").app
        print("✅ Servidor FastAPI importado correctamente")
        
        _ = __import__("src.api.client").MitsubaAPIClient
        print("✅ Cliente API importado correctamente")
        
        from src.config import get_config
        _ = get_config()
        print("✅ Configuración cargada correctamente")
        
        return True
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 Iniciando pruebas del sistema Mitsuba Scene Viewer")
    print("=" * 60)
    
    # Test 1: Imports
    print("\n1️⃣ Probando imports...")
    if not test_imports():
        print("❌ Falló la prueba de imports")
        return False
    
    # Test 2: Conexión al servidor
    print("\n2️⃣ Probando conexión al servidor...")
    if not test_server_connection():
        print("⚠️  Servidor no está ejecutándose. Para probarlo:")
        print("   python run_server.py")
        print("   Luego ejecuta este script nuevamente")
        return False
    
    # Test 3: Endpoints API
    print("\n3️⃣ Probando endpoints de la API...")
    if not test_api_endpoints():
        print("❌ Falló la prueba de endpoints")
        return False
    
    print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    print("\n📋 Para usar el sistema:")
    print("   1. Ejecutar servidor: python run_server.py")
    print("   2. Ejecutar cliente: python run_client.py")
    print("   3. Abrir navegador en: http://localhost:7860")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
