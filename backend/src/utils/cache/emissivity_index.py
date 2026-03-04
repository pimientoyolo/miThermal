"""
Índice en memoria para búsqueda rápida de archivos de emisividad.

Optimiza la búsqueda de O(n) a O(1) construyendo un índice una sola vez al iniciar.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from src.config import OUTPUT_STATIC_DIR

logger = logging.getLogger(__name__)


class EmissivityFileIndex:
    """
    Índice en memoria para búsqueda rápida de archivos de emisividad.
    
    Escanea OUTPUT_STATIC_DIR una sola vez al iniciar y construye un índice
    {object_id: file_path} para búsqueda O(1).
    """
    
    def __init__(self):
        self.index: Dict[str, Path] = {}
        self._build_index()
    
    def _build_index(self):
        """Construye el índice escaneando el directorio de salida."""
        try:
            logger.info("Construyendo índice de archivos de emisividad...")
            base_dir = Path(OUTPUT_STATIC_DIR)
            
            if not base_dir.exists():
                logger.warning(f"Directorio {base_dir} no existe")
                return
            
            # Escanear archivos .txt recursivamente
            for txt_file in base_dir.glob("**/*.txt"):
                # Obtener la ruta relativa al directorio base
                try:
                    relative_path = txt_file.relative_to(base_dir)
                    # Usar la ruta sin extensión como key
                    key = str(relative_path.with_suffix(''))
                    self.index[key] = txt_file
                    
                    # También indexar por nombre de archivo solo (sin directorios)
                    self.index[txt_file.stem] = txt_file
                except ValueError:
                    continue
            
            logger.info(f"Índice construido: {len(self.index)} archivos indexados")
            
        except Exception as e:
            logger.error(f"Error construyendo índice de emisividad: {e}")
    
    def get(self, object_id: str) -> Optional[Path]:
        """
        Busca un archivo de emisividad por object_id.
        
        Args:
            object_id: ID del objeto (puede incluir rutas)
            
        Returns:
            Path al archivo o None si no se encuentra
        """
        # Limpiar el object_id
        clean_id = object_id.replace('\\', '/').strip('/')
        
        # Intentar varias variaciones
        candidates = [
            clean_id,  # Ruta completa
            Path(clean_id).stem,  # Solo nombre sin extensión
            Path(clean_id).name,  # Con extensión
        ]
        
        for candidate in candidates:
            if candidate in self.index:
                return self.index[candidate]
        
        return None
    
    def refresh(self):
        """Reconstruye el índice (útil si se agregan archivos nuevos)."""
        self.index.clear()
        self._build_index()
    
    def get_all_files(self) -> list[str]:
        """Retorna lista de todos los archivos indexados."""
        return list(self.index.keys())
    
    def size(self) -> int:
        """Retorna el número de archivos indexados."""
        return len(self.index)


# Instancia global (singleton)
_emissivity_index: Optional[EmissivityFileIndex] = None


def get_emissivity_index() -> EmissivityFileIndex:
    """Obtiene la instancia del índice (singleton con lazy loading)."""
    global _emissivity_index
    if _emissivity_index is None:
        _emissivity_index = EmissivityFileIndex()
    return _emissivity_index
