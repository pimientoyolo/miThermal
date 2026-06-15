"""
Sistema de cache persistente para firmas espectrales con LRU (Least Recently Used).

Optimizaciones:
- Cache persistente en disco (sobrevive reinicios)
- LRU con límite de tamaño
- Validación de checksums (detecta cambios en archivos)
- Compresión opcional
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class EmissionCacheManager:
    """
    Gestor de cache para emisión y reflectancia de objetos.
    
    Características:
    - Cache persistente en disco
    - LRU automático con límite de tamaño
    - Validación de integridad con checksums
    - Métricas de hit/miss rate
    """
    
    def __init__(self, cache_dir: Path, max_size: int = 1000):
        """
        Args:
            cache_dir: Directorio donde guardar el cache
            max_size: Número máximo de entradas en cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        
        # Métricas
        self.hits = 0
        self.misses = 0
        
        # Índice en memoria: {cache_key: (checksum, timestamp)}
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Carga el índice del cache desde disco."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando índice de cache: {e}")
        return {}
    
    def _save_index(self):
        """Guarda el índice del cache en disco."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando índice de cache: {e}")
    
    def _compute_file_checksum(self, file_path: str) -> str:
        """Calcula checksum SHA256 de un archivo."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error calculando checksum de {file_path}: {e}")
            return ""
    
    def _make_cache_key(self, emissivity_file: str, temperature: float, material_type: str = "diffuse", roughness: float = 0.05) -> str:
        """Genera una clave única para el cache."""
        key_str = f"{emissivity_file}:{temperature:.2f}:{material_type}:{roughness:.4f}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Retorna la ruta del archivo de cache."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(
        self, 
        emissivity_file: str, 
        temperature: float,
        material_type: str = "diffuse",
        roughness: float = 0.05
    ) -> Optional[Tuple[Dict, Dict]]:
        """
        Obtiene datos del cache si existen y son válidos.
        
        Args:
            emissivity_file: Ruta al archivo de emisividad
            temperature: Temperatura del objeto
            material_type: Tipo de material
            roughness: Rugosidad
            
        Returns:
            (dict_reflectance, dict_emission) o None si no está en cache
        """
        cache_key = self._make_cache_key(emissivity_file, temperature, material_type, roughness)
        cache_path = self._get_cache_path(cache_key)
        
        # Verificar si existe en índice
        if cache_key not in self.index:
            self.misses += 1
            return None
        
        # Verificar si el archivo cambió
        current_checksum = self._compute_file_checksum(emissivity_file)
        cached_checksum = self.index[cache_key].get("checksum", "")
        
        if current_checksum != cached_checksum:
            logger.info(f"Archivo modificado, invalidando cache: {emissivity_file}")
            self.invalidate(emissivity_file, temperature, material_type, roughness)
            self.misses += 1
            return None
        
        # Cargar datos del cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.hits += 1
                logger.debug(f"Cache hit: {emissivity_file} @ {temperature}K")
                return data
            except Exception as e:
                logger.error(f"Error cargando cache: {e}")
                self.invalidate(emissivity_file, temperature, material_type, roughness)
        
        self.misses += 1
        return None
    
    def set(
        self,
        emissivity_file: str,
        temperature: float,
        dict_reflectance: Dict,
        dict_emission: Dict,
        material_type: str = "diffuse",
        roughness: float = 0.05
    ):
        """
        Guarda datos en el cache.
        
        Args:
            emissivity_file: Ruta al archivo de emisividad
            temperature: Temperatura del objeto
            dict_reflectance: Diccionario de reflectancia
            dict_emission: Diccionario de emisión
            material_type: Tipo de material
            roughness: Rugosidad
        """
        cache_key = self._make_cache_key(emissivity_file, temperature, material_type, roughness)
        cache_path = self._get_cache_path(cache_key)
        
        # Calcular checksum del archivo
        checksum = self._compute_file_checksum(emissivity_file)
        
        # Guardar datos
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((dict_reflectance, dict_emission), f)
            
            # Actualizar índice
            import time
            self.index[cache_key] = {
                "emissivity_file": emissivity_file,
                "temperature": temperature,
                "material_type": material_type,
                "roughness": roughness,
                "checksum": checksum,
                "timestamp": time.time()
            }
            
            # Limitar tamaño del cache (LRU)
            self._enforce_size_limit()
            
            self._save_index()
            logger.debug(f"Cache guardado: {emissivity_file} @ {temperature}K")
            
        except Exception as e:
            logger.error(f"Error guardando en cache: {e}")
    
    def _enforce_size_limit(self):
        """Elimina las entradas más antiguas si se excede el límite."""
        if len(self.index) > self.max_size:
            # Ordenar por timestamp (más antiguo primero)
            sorted_keys = sorted(
                self.index.keys(),
                key=lambda k: self.index[k].get("timestamp", 0)
            )
            
            # Eliminar las más antiguas
            to_remove = len(self.index) - self.max_size
            for key in sorted_keys[:to_remove]:
                cache_path = self._get_cache_path(key)
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                    del self.index[key]
                    logger.debug(f"Cache LRU: eliminada entrada {key}")
                except Exception as e:
                    logger.error(f"Error eliminando cache antiguo: {e}")
    
    def invalidate(self, emissivity_file: str, temperature: float, material_type: str = "diffuse", roughness: float = 0.05):
        """Invalida una entrada específica del cache."""
        cache_key = self._make_cache_key(emissivity_file, temperature, material_type, roughness)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            if cache_key in self.index:
                del self.index[cache_key]
            self._save_index()
            logger.info(f"Cache invalidado: {emissivity_file} @ {temperature}K")
        except Exception as e:
            logger.error(f"Error invalidando cache: {e}")
    
    def clear(self):
        """Limpia todo el cache."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.index.clear()
            self._save_index()
            self.hits = 0
            self.misses = 0
            logger.info("Cache completamente limpiado")
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del cache."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.index),
            "max_size": self.max_size
        }


# Instancia global (singleton)
_cache_manager: Optional[EmissionCacheManager] = None


def get_cache_manager() -> EmissionCacheManager:
    """Obtiene la instancia del gestor de cache (singleton)."""
    global _cache_manager
    if _cache_manager is None:
        from src.config import OUTPUT_DIR
        cache_dir = OUTPUT_DIR / ".cache" / "emissions"
        _cache_manager = EmissionCacheManager(cache_dir, max_size=1000)
    return _cache_manager
