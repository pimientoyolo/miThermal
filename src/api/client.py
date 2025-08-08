"""
Cliente web para API de visualización de escenas Mitsuba con Gradio
"""
import gradio as gr
import requests
import json
from pathlib import Path
from typing import Dict
import tempfile
import logging

logger = logging.getLogger(__name__)

# Variables globales
api_client = None
object_id_mapping = {}  # Mapeo de opciones del dropdown con IDs de objetos

class MitsubaAPIClient:
    """Cliente para conectar con la API FastAPI del servidor Mitsuba"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Verifica si el servidor está disponible"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error conectando al servidor: {e}")
            return False
    
    def upload_scene(self, xml_file_path: str) -> Dict:
        """Sube un archivo XML de escena al servidor"""
        try:
            with open(xml_file_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/upload-scene", files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error subiendo escena: {e}")
            return {"status": "error", "detail": str(e)}
    
    def load_scene(self, scene_path: str) -> Dict:
        """Carga una escena desde una ruta del servidor"""
        try:
            response = self.session.post(
                f"{self.base_url}/load-scene",
                params={"scene_path": scene_path}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error cargando escena: {e}")
            return {"status": "error", "detail": str(e)}
    
    def get_objects(self) -> Dict:
        """Obtiene la lista de objetos de la escena actual"""
        try:
            response = self.session.get(f"{self.base_url}/objects")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo objetos: {e}")
            return {"status": "error", "detail": str(e)}
    
    def get_object(self, object_id: str) -> Dict:
        """Obtiene información de un objeto específico"""
        try:
            response = self.session.get(f"{self.base_url}/objects/{object_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo objeto {object_id}: {e}")
            return {"status": "error", "detail": str(e)}
    
    def download_object(self, object_id: str, save_path: str) -> str:
        """Descarga el archivo de un objeto específico y retorna la ruta real"""
        try:
            response = self.session.get(f"{self.base_url}/download/{object_id}")
            response.raise_for_status()
            
            # Obtener información del objeto para determinar el tipo de archivo
            obj_info = self.get_object(object_id)
            original_ext = ".obj"  # Default
            
            if obj_info.get("status") == "success":
                original_filename = obj_info["object"].get("filename", "")
                if original_filename:
                    original_ext = Path(original_filename).suffix.lower()
            
            # Ajustar la ruta de guardado para mantener la extensión original
            save_path = Path(save_path)
            if save_path.suffix.lower() != original_ext:
                save_path = save_path.with_suffix(original_ext)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Archivo descargado: {save_path} (tipo: {original_ext})")
            return str(save_path)  # Devolver la ruta real con extensión correcta
            
        except Exception as e:
            logger.error(f"Error descargando objeto {object_id}: {e}")
            return ""
    
    def get_scene_info(self) -> Dict:
        """Obtiene información general de la escena"""
        try:
            response = self.session.get(f"{self.base_url}/scene-info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo info de escena: {e}")
            return {"status": "error", "detail": str(e)}

# Cliente global
api_client = MitsubaAPIClient()

def check_server_status():
    """Verifica el estado del servidor"""
    if api_client.health_check():
        return "✅ Servidor conectado", "success"
    else:
        return "❌ Servidor no disponible", "error"

def upload_xml_file(file_obj):
    """Maneja la subida de archivos XML"""
    global object_id_mapping
    
    if file_obj is None:
        return "❌ No se seleccionó archivo", gr.Dropdown(choices=[]), ""
    
    try:
        # Subir archivo al servidor
        result = api_client.upload_scene(file_obj.name)
        
        if result.get("status") == "success":
            # Obtener lista de objetos
            objects_data = api_client.get_objects()
            if objects_data.get("status") == "success":
                objects = objects_data.get("objects", [])
                
                # Formatear información para mostrar
                info_text = "✅ Escena cargada exitosamente\n"
                info_text += f"📁 Total de objetos: {len(objects)}\n\n"
                
                # Crear lista de opciones para el dropdown y mapeo
                object_choices = []
                object_id_mapping = {}
                
                for obj in objects:
                    obj_info = f"{obj['id']} ({obj['type']})"
                    if obj.get('has_material'):
                        obj_info += " 🎨"
                    if obj.get('has_emission'):
                        obj_info += " 💡"
                    
                    object_choices.append(obj_info)
                    object_id_mapping[obj_info] = obj['id']
                
                return info_text, gr.Dropdown(choices=object_choices), json.dumps(objects, indent=2)
            else:
                return f"❌ Error obteniendo objetos: {objects_data.get('detail', 'Error desconocido')}", gr.Dropdown(choices=[]), ""
        else:
            return f"❌ Error subiendo archivo: {result.get('detail', 'Error desconocido')}", gr.Dropdown(choices=[]), ""
            
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.Dropdown(choices=[]), ""

def load_server_scene(scene_path):
    """Carga una escena desde el servidor"""
    global object_id_mapping
    
    if not scene_path.strip():
        return "❌ Ingrese una ruta válida", gr.Dropdown(choices=[]), ""
    
    try:
        result = api_client.load_scene(scene_path.strip())
        
        if result.get("status") == "success":
            # Obtener lista de objetos
            objects_data = api_client.get_objects()
            if objects_data.get("status") == "success":
                objects = objects_data.get("objects", [])
                
                info_text = "✅ Escena cargada desde servidor\n"
                info_text += f"📁 Total de objetos: {len(objects)}\n\n"
                
                # Crear lista de opciones para el dropdown y mapeo
                object_choices = []
                object_id_mapping = {}
                
                for obj in objects:
                    obj_info = f"{obj['id']} ({obj['type']})"
                    if obj.get('has_material'):
                        obj_info += " 🎨"
                    if obj.get('has_emission'):
                        obj_info += " 💡"
                    
                    object_choices.append(obj_info)
                    object_id_mapping[obj_info] = obj['id']
                
                return info_text, gr.Dropdown(choices=object_choices), json.dumps(objects, indent=2)
            else:
                return f"❌ Error obteniendo objetos: {objects_data.get('detail', 'Error desconocido')}", gr.Dropdown(choices=[]), ""
        else:
            return f"❌ Error cargando escena: {result.get('detail', 'Error desconocido')}", gr.Dropdown(choices=[]), ""
            
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.Dropdown(choices=[]), ""

def view_object_3d(object_choice):
    """Visualiza un objeto 3D específico"""
    if not object_choice:
        return None, "Seleccione un objeto para visualizar"
    
    # Obtener el ID del objeto desde el mapeo
    object_id = object_id_mapping.get(object_choice)
    if not object_id:
        return None, "❌ Error: ID de objeto no encontrado"
    
    try:
        # Obtener información del objeto
        obj_data = api_client.get_object(object_id)
        
        if obj_data.get("status") == "success":
            obj = obj_data.get("object", {})
            
            # Descargar archivo del modelo
            temp_dir = tempfile.mkdtemp()
            temp_file_base = Path(temp_dir) / f"{object_id}"
            
            # Descargar con la extensión correcta
            downloaded_file = api_client.download_object(object_id, str(temp_file_base))
            
            if downloaded_file:
                downloaded_path = Path(downloaded_file)
                file_ext = downloaded_path.suffix.lower()
                
                # Formatear información del objeto
                info_text = f"🎯 Objeto: {obj['id']}\n"
                info_text += f"📦 Tipo: {obj['type']}\n"
                info_text += f"📄 Archivo: {obj['filename']}\n"
                info_text += f"🔧 Formato descargado: {file_ext}\n"
                info_text += f"💾 Tamaño: {downloaded_path.stat().st_size} bytes\n"
                
                if obj.get('transform'):
                    info_text += f"🔄 Transformaciones: {len(obj['transform'])} definidas\n"
                
                if obj.get('has_material'):
                    info_text += f"🎨 Material: {obj['bsdf'].get('type', 'Desconocido')}\n"
                
                if obj.get('has_emission'):
                    info_text += f"💡 Emisor: {obj['emitter'].get('type', 'Desconocido')}\n"
                
                # Verificar si es un archivo PLY binario y advertir
                if file_ext == ".ply":
                    info_text += "\n⚠️ ARCHIVO PLY DETECTADO\n"
                    info_text += "Gradio funciona mejor con archivos OBJ.\n"
                    
                    # Intentar leer las primeras líneas para ver si es texto o binario
                    try:
                        with open(downloaded_path, 'rb') as f:
                            header = f.read(200)
                            if b'format binary' in header:
                                info_text += "🔍 Formato: PLY binario\n"
                                info_text += "💡 Sugerencia: Convertir a OBJ con MeshLab o Blender\n"
                                # El archivo binario puede no renderizarse correctamente
                                # Podrías retornar None aquí si quieres evitar errores
                            else:
                                info_text += "🔍 Formato: PLY ASCII (debería funcionar)\n"
                    except Exception:
                        pass
                elif file_ext not in [".obj", ".ply"]:
                    info_text += f"\n⚠️ Formato {file_ext} puede no ser compatible con el visualizador\n"
                
                return str(downloaded_path), info_text
            else:
                return None, f"❌ Error descargando archivo del objeto {object_id}"
        else:
            return None, f"❌ Error obteniendo objeto: {obj_data.get('detail', 'Error desconocido')}"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def create_mitsuba_viewer_interface():
    """Crea la interfaz principal del visualizador"""
    global api_client
    
    # Inicializar cliente API si no existe
    if api_client is None:
        api_client = MitsubaAPIClient()
    
    with gr.Blocks(
        title="Mitsuba Scene Viewer",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1600px !important;
        }
        .model-viewer {
            height: 600px;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <h1 style="text-align: center; color: #2e86de;">
            🎨 Visualizador de Escenas Mitsuba 3D
        </h1>
        <p style="text-align: center; font-size: 18px;">
            Cliente web para visualización y análisis de objetos 3D en escenas Mitsuba
        </p>
        """)
        
        # Estado del servidor
        with gr.Row():
            with gr.Column(scale=1):
                server_status = gr.Textbox(
                    label="Estado del Servidor",
                    value="Verificando...",
                    interactive=False
                )
                check_btn = gr.Button("🔄 Verificar Conexión", variant="secondary")
        
        with gr.Tabs():
            # Tab 1: Cargar Escena
            with gr.Tab("📁 Cargar Escena"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>Subir archivo XML</h3>")
                        xml_file = gr.File(
                            label="Archivo XML de Mitsuba",
                            file_types=[".xml"]
                        )
                        upload_btn = gr.Button("📤 Subir Escena", variant="primary")
                        
                        gr.HTML("<h3>O cargar desde servidor</h3>")
                        server_path = gr.Textbox(
                            label="Ruta en el servidor",
                            placeholder="/ruta/a/escena.xml",
                            lines=1
                        )
                        load_btn = gr.Button("📥 Cargar del Servidor", variant="secondary")
                    
                    with gr.Column(scale=2):
                        scene_info = gr.Textbox(
                            label="Información de la Escena",
                            lines=10,
                            interactive=False
                        )
                        
                        scene_json = gr.JSON(
                            label="Datos de Objetos (JSON)",
                            visible=False
                        )
            
            # Tab 2: Visualizar Objetos
            with gr.Tab("🎯 Visualizar Objetos 3D"):
                with gr.Row():
                    with gr.Column(scale=1):
                        object_selector = gr.Dropdown(
                            label="Seleccionar Objeto",
                            choices=[],
                            interactive=True
                        )
                        
                        view_btn = gr.Button("👁️ Visualizar Objeto", variant="primary")
                        
                        object_info = gr.Textbox(
                            label="Información del Objeto",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        model_viewer = gr.Model3D(
                            label="Visualizador 3D",
                            height=600
                        )
            
            # Tab 3: Análisis de Escena
            with gr.Tab("📊 Análisis de Escena"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3>Estadísticas de la Escena</h3>")
                        
                        # Componente preparado para futuras estadísticas
                        _ = gr.Textbox(
                            label="Estadísticas",
                            lines=10,
                            interactive=False
                        )
                        
                        # Botón preparado para actualizar estadísticas
                        _ = gr.Button("🔄 Actualizar Estadísticas")
                    
                    with gr.Column():
                        gr.HTML("<h3>Información Detallada</h3>")
                        
                        # Componente preparado para información detallada
                        _ = gr.JSON(
                            label="Información Completa de la Escena"
                        )
        
        # Configurar eventos
        check_btn.click(
            fn=check_server_status,
            outputs=[server_status]
        )
        
        upload_btn.click(
            fn=upload_xml_file,
            inputs=[xml_file],
            outputs=[scene_info, object_selector, scene_json]
        )
        
        load_btn.click(
            fn=load_server_scene,
            inputs=[server_path],
            outputs=[scene_info, object_selector, scene_json]
        )
        
        view_btn.click(
            fn=view_object_3d,
            inputs=[object_selector],
            outputs=[model_viewer, object_info]
        )
        
        # Verificar estado inicial del servidor
        interface.load(
            fn=check_server_status,
            outputs=[server_status]
        )
    
    return interface

def launch_client(server_port: int = 7860, server_host: str = "0.0.0.0", 
                 api_url: str = "http://localhost:8000"):
    """
    Lanza el cliente Gradio
    
    Args:
        server_port (int): Puerto del cliente Gradio
        server_host (str): Host del cliente Gradio
        api_url (str): URL de la API FastAPI
    """
    global api_client
    api_client = MitsubaAPIClient(api_url)
    
    interface = create_mitsuba_viewer_interface()
    
    interface.launch(
        server_name=server_host,
        server_port=server_port,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    launch_client()
