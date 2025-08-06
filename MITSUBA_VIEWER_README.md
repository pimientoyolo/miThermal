# Mitsuba Scene Viewer - Sistema Cliente-Servidor

Sistema modular para visualización e inspección de escenas XML de Mitsuba con arquitectura cliente-servidor.

## 🏗️ Arquitectura

```
├── 🖥️  Servidor (FastAPI)          ├── 🌐 Cliente (Gradio)
│   ├── Parseo XML Mitsuba         │   ├── Interfaz web intuitiva
│   ├── Extracción objetos 3D      │   ├── Carga/subida archivos XML
│   ├── API REST endpoints         │   ├── Lista y selección objetos
│   └── Servicio archivos mesh     │   └── Visualización 3D interactiva
```

## ✨ Funcionalidades

### Servidor (FastAPI)
- ✅ **Parseo XML Mitsuba**: Extrae objetos 3D, materiales, emisores y transformaciones
- ✅ **API REST**: Endpoints para subir/cargar escenas, listar objetos y descargar meshes
- ✅ **Servicio estático**: Sirve archivos OBJ/PLY para visualización
- ✅ **CORS configurado**: Permite acceso desde cliente web
- 🔄 **Preparado para edición**: Endpoints base para futuras modificaciones de propiedades

### Cliente (Gradio)
- ✅ **Interfaz web moderna**: Subida de archivos XML drag & drop
- ✅ **Carga desde servidor**: Selección de escenas desde rutas del servidor
- ✅ **Lista objetos**: Visualización tabular con filtros y selección
- ✅ **Visualización 3D**: Renderizado interactivo con gr.Model3D
- ✅ **Información detallada**: Propiedades de objetos y escena
- 🔄 **Preparado para edición**: UI base para futuras modificaciones

## 🚀 Uso Rápido

### 1. Instalar dependencias
```bash
pip install fastapi uvicorn gradio requests pathlib
```

### 2. Lanzar servidor
```bash
python run_server.py
```
- Servidor: http://localhost:8000
- Documentación: http://localhost:8000/docs

### 3. Lanzar cliente (en otra terminal)
```bash
python run_client.py
```
- Cliente: http://localhost:7860

### 4. Usar la interfaz
1. **Verificar conexión**: Clic en "🔍 Verificar Servidor"
2. **Cargar escena**:
   - **Opción A**: Subir archivo XML local
   - **Opción B**: Cargar desde ruta del servidor
3. **Explorar objetos**: Seleccionar objetos de la tabla
4. **Visualizar 3D**: Ver modelos en el visor interactivo

## 📁 Estructura de Archivos

```
src/
├── api/
│   ├── __init__.py
│   ├── server.py           # Servidor FastAPI
│   └── client.py           # Cliente Gradio
├── mitsuba_core/
│   └── scene_parser.py     # Parser XML Mitsuba
├── config.py               # Configuración
└── ...

run_server.py               # Script servidor
run_client.py               # Script cliente
```

## 🔌 API Endpoints

### Servidor FastAPI

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servidor |
| `/upload-scene` | POST | Subir archivo XML |
| `/load-scene` | POST | Cargar escena desde ruta |
| `/objects` | GET | Listar todos los objetos |
| `/objects/{id}` | GET | Información de objeto específico |
| `/objects/{id}/download` | GET | Descargar archivo mesh |
| `/objects/{id}/update` | PUT | Actualizar propiedades (preparado) |
| `/scene-info` | GET | Información general de la escena |

### Ejemplo uso API
```python
import requests

# Subir escena
with open("escena.xml", "rb") as f:
    response = requests.post("http://localhost:8000/upload-scene", files={"file": f})

# Listar objetos
objects = requests.get("http://localhost:8000/objects").json()

# Descargar mesh
mesh_file = requests.get(f"http://localhost:8000/objects/{obj_id}/download")
```

## 🔮 Extensiones Futuras

### ✅ Implementado
- Parser XML completo con extracción de objetos 3D
- API REST funcional con todos los endpoints base
- Cliente web con visualización 3D
- Arquitectura modular y extensible

### 🔄 Preparado para implementar
- **Edición de propiedades**: Modificar materiales, transforms, emisores
- **Análisis de escena**: Estadísticas y métricas detalladas
- **Formatos adicionales**: Soporte PLY y otros formatos mesh
- **Autenticación**: Sistema de usuarios y permisos
- **Optimización**: Cache, compresión, streaming

### 💡 Ideas futuras
- **Editor visual**: Drag & drop para modificar escenas
- **Renderizado en tiempo real**: Preview de cambios
- **Colaboración**: Edición multi-usuario
- **Plugins**: Sistema de extensiones personalizadas

## 🛠️ Desarrollo

### Estructura modular
```python
# Servidor
from src.api.server import app  # FastAPI app
from src.mitsuba_core.scene_parser import MitsubaSceneParser

# Cliente  
from src.api.client import MitsubaAPIClient, create_mitsuba_interface
```

### Agregar nuevos endpoints
```python
# En server.py
@app.post("/nuevo-endpoint")
async def nuevo_endpoint():
    # Tu lógica aquí
    return {"status": "success"}
```

### Extender cliente
```python
# En client.py - agregar nuevas funciones
def nueva_funcionalidad():
    # Tu lógica aquí
    pass

# Agregar a la interfaz Gradio
with gr.Tab("Nueva Funcionalidad"):
    # Componentes UI
    pass
```

## 📋 Notas Técnicas

- **CORS**: Configurado para desarrollo (`allow_origins=["*"]`)
- **Archivos estáticos**: Servidos desde `/static` 
- **Resolución de rutas**: Automática relativa al XML
- **Manejo de errores**: Logging completo y respuestas informativas
- **Formato mesh**: Principalmente OBJ, PLY en desarrollo

## 🐛 Troubleshooting

### "Servidor no disponible"
1. Verificar que el servidor esté ejecutándose en puerto 8000
2. Comprobar firewall/permisos de red
3. Revisar logs del servidor

### "Archivo mesh no encontrado"
1. Verificar rutas relativas en XML
2. Comprobar permisos de archivos
3. Revisar logs del parser

### "Error de visualización 3D"
1. Verificar formato del archivo mesh (OBJ recomendado)
2. Comprobar tamaño del archivo (límites de Gradio)
3. Probar con otro navegador

---

**Sistema desarrollado para análisis y visualización de escenas Mitsuba**  
🔗 Arquitectura modular • 🎨 UI moderna • 🚀 Listo para producción
