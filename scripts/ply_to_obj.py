#!/usr/bin/env python3
"""
Utilidad para convertir archivos PLY a OBJ para mejor compatibilidad con Gradio
"""
import sys
from pathlib import Path

def convert_ply_to_obj(ply_path, obj_path):
    """
    Convierte un archivo PLY simple a OBJ
    Soporta solo PLY ASCII con vértices y caras básicas
    """
    try:
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        # Encontrar el número de vértices y caras
        vertices = []
        faces = []
        reading_vertices = False
        reading_faces = False
        vertex_count = 0
        face_count = 0
        current_vertex = 0
        current_face = 0
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                continue
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
                continue
            elif line == 'end_header':
                reading_vertices = True
                continue
            
            if reading_vertices and current_vertex < vertex_count:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = parts[0], parts[1], parts[2]
                    vertices.append(f"v {x} {y} {z}\n")
                    current_vertex += 1
                    
                if current_vertex >= vertex_count:
                    reading_vertices = False
                    reading_faces = True
                    
            elif reading_faces and current_face < face_count:
                parts = line.split()
                if len(parts) >= 4:  # Al menos 3 índices + contador
                    num_vertices = int(parts[0])
                    if num_vertices == 3 and len(parts) >= 4:
                        # Cara triangular
                        i1, i2, i3 = int(parts[1]) + 1, int(parts[2]) + 1, int(parts[3]) + 1
                        faces.append(f"f {i1} {i2} {i3}\n")
                    elif num_vertices == 4 and len(parts) >= 5:
                        # Cara cuadrangular
                        i1, i2, i3, i4 = int(parts[1]) + 1, int(parts[2]) + 1, int(parts[3]) + 1, int(parts[4]) + 1
                        faces.append(f"f {i1} {i2} {i3} {i4}\n")
                    current_face += 1
        
        # Escribir archivo OBJ
        with open(obj_path, 'w') as f:
            f.write("# Convertido de PLY a OBJ\n")
            f.writelines(vertices)
            f.writelines(faces)
        
        return True
        
    except Exception as e:
        print(f"Error convirtiendo PLY a OBJ: {e}")
        return False

def check_ply_format(ply_path):
    """Verifica el formato de un archivo PLY"""
    try:
        with open(ply_path, 'rb') as f:
            header = f.read(200).decode('utf-8', errors='ignore')
            
        if 'format ascii' in header:
            return "ASCII"
        elif 'format binary' in header:
            return "Binario"
        else:
            return "Desconocido"
    except Exception:
        return "Error"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python ply_to_obj.py archivo.ply archivo.obj")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    obj_file = sys.argv[2]
    
    if not Path(ply_file).exists():
        print(f"Error: {ply_file} no existe")
        sys.exit(1)
    
    print(f"Convirtiendo {ply_file} a {obj_file}...")
    formato = check_ply_format(ply_file)
    print(f"Formato PLY detectado: {formato}")
    
    if formato == "Binario":
        print("⚠️ Archivo PLY binario detectado. Esta herramienta solo soporta PLY ASCII.")
        print("Para convertir PLY binario, usa herramientas como MeshLab o Blender.")
        sys.exit(1)
    
    if convert_ply_to_obj(ply_file, obj_file):
        print(f"✅ Conversión exitosa: {obj_file}")
    else:
        print("❌ Error en la conversión")
        sys.exit(1)
