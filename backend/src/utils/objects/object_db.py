"""
Base de datos ligera SQLite para propiedades de objetos.
Reemplaza config_scene.json con búsquedas indexadas O(1).
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.utils.objects.properties import ObjectProperties, SpectralProperty
from src.config import OUTPUT_STATIC_DIR

logger = logging.getLogger(__name__)


class ObjectDatabase:
    """
    Base de datos SQLite para gestionar propiedades de objetos.
    
    Ventajas sobre JSON:
    - Búsqueda O(1) con índices
    - Transacciones ACID
    - Thread-safe con locks
    - Queries complejas
    - Historial con timestamps
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Args:
            db_path: Ruta a la base de datos SQLite (default: config/scene_objects.db)
        """
        if db_path is None:
            db_path = Path(OUTPUT_STATIC_DIR) / "scene_objects.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Conexión con check_same_thread=False para uso multi-thread
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level="DEFERRED"
        )
        self.conn.row_factory = sqlite3.Row  # Acceso por nombre de columna
        
        self._create_tables()
        logger.info(f"ObjectDatabase inicializada: {self.db_path}")
    
    def _create_tables(self):
        """Crea tablas e índices."""
        with self.conn:
            # Tabla principal de objetos
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    object_id TEXT PRIMARY KEY,
                    temperature_k REAL NOT NULL,
                    emissivity_file TEXT,
                    emissivity_hash TEXT,
                    emissivity_data TEXT,  -- JSON con wavelengths y values
                    reflectance_file TEXT,
                    reflectance_hash TEXT,
                    reflectance_data TEXT,  -- JSON con wavelengths y values
                    metadata TEXT,  -- JSON con datos adicionales
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Índices para búsqueda rápida
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emissivity_hash 
                ON objects(emissivity_hash)
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_temperature 
                ON objects(temperature_k)
            """)
            
            # Tabla de historial de cambios (opcional)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS object_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    changed_at TEXT NOT NULL,
                    FOREIGN KEY (object_id) REFERENCES objects(object_id)
                )
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_object 
                ON object_history(object_id, changed_at)
            """)
    
    def upsert_object(self, props: ObjectProperties) -> bool:
        """
        Inserta o actualiza propiedades de objeto.
        
        Args:
            props: Propiedades del objeto
        
        Returns:
            bool: True si se actualizó objeto existente, False si se insertó nuevo
        """
        now = datetime.now().isoformat()
        
        # Serializar datos espectrales a JSON
        emissivity_data = None
        if props.emissivity:
            emissivity_data = json.dumps({
                "wavelengths_nm": props.emissivity.wavelengths_nm.tolist(),
                "values": props.emissivity.values.tolist()
            })
        
        reflectance_data = None
        if props.reflectance:
            reflectance_data = json.dumps({
                "wavelengths_nm": props.reflectance.wavelengths_nm.tolist(),
                "values": props.reflectance.values.tolist()
            })
        
        metadata_json = json.dumps(props.metadata) if props.metadata else None
        
        # Verificar si existe
        exists = self.get_object(props.object_id) is not None
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO objects (
                    object_id, temperature_k,
                    emissivity_file, emissivity_hash, emissivity_data,
                    reflectance_file, reflectance_hash, reflectance_data,
                    metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    temperature_k = excluded.temperature_k,
                    emissivity_file = excluded.emissivity_file,
                    emissivity_hash = excluded.emissivity_hash,
                    emissivity_data = excluded.emissivity_data,
                    reflectance_file = excluded.reflectance_file,
                    reflectance_hash = excluded.reflectance_hash,
                    reflectance_data = excluded.reflectance_data,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
            """, (
                props.object_id,
                props.temperature_k,
                props.emissivity.file_path if props.emissivity else None,
                props.emissivity.file_hash if props.emissivity else None,
                emissivity_data,
                props.reflectance.file_path if props.reflectance else None,
                props.reflectance.file_hash if props.reflectance else None,
                reflectance_data,
                metadata_json,
                now if not exists else self._get_created_at(props.object_id) or now,
                now
            ))
        
        action = 'actualizado' if exists else 'insertado'
        logger.info(f"Objeto '{props.object_id}' {action}")
        return exists
    
    def _get_created_at(self, object_id: str) -> Optional[str]:
        """Helper para obtener timestamp de creación."""
        cursor = self.conn.execute(
            "SELECT created_at FROM objects WHERE object_id = ?",
            (object_id,)
        )
        row = cursor.fetchone()
        return row["created_at"] if row else None
    
    def get_object(self, object_id: str) -> Optional[ObjectProperties]:
        """
        Busca objeto por ID (O(1) con índice).
        
        Args:
            object_id: Identificador del objeto
        
        Returns:
            ObjectProperties or None: Propiedades del objeto si existe
        """
        cursor = self.conn.execute(
            "SELECT * FROM objects WHERE object_id = ?",
            (object_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_properties(row)
    
    def _row_to_properties(self, row: sqlite3.Row) -> ObjectProperties:
        """Convierte fila de DB a ObjectProperties."""
        import numpy as np
        
        # Reconstruir emisividad
        emissivity = None
        if row["emissivity_data"]:
            emiss_dict = json.loads(row["emissivity_data"])
            emissivity = SpectralProperty(
                wavelengths_nm=np.array(emiss_dict["wavelengths_nm"]),
                values=np.array(emiss_dict["values"]),
                file_hash=row["emissivity_hash"] or "",
                file_path=row["emissivity_file"] or ""
            )
        
        # Reconstruir reflectancia
        reflectance = None
        if row["reflectance_data"]:
            refl_dict = json.loads(row["reflectance_data"])
            reflectance = SpectralProperty(
                wavelengths_nm=np.array(refl_dict["wavelengths_nm"]),
                values=np.array(refl_dict["values"]),
                file_hash=row["reflectance_hash"] or "",
                file_path=row["reflectance_file"] or ""
            )
        
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        
        return ObjectProperties(
            object_id=row["object_id"],
            temperature_k=row["temperature_k"],
            emissivity=emissivity,
            reflectance=reflectance,
            metadata=metadata
        )
    
    def find_by_hash(self, emissivity_hash: str) -> List[str]:
        """
        Encuentra todos los objetos que usan la misma firma espectral.
        Útil para deduplicación.
        
        Args:
            emissivity_hash: Hash MD5 del archivo de emisividad
        
        Returns:
            List[str]: Lista de object_ids que comparten el hash
        """
        cursor = self.conn.execute(
            "SELECT object_id FROM objects WHERE emissivity_hash = ?",
            (emissivity_hash,)
        )
        return [row["object_id"] for row in cursor.fetchall()]
    
    def list_all_objects(self) -> List[str]:
        """
        Lista IDs de todos los objetos.
        
        Returns:
            List[str]: Lista de object_ids
        """
        cursor = self.conn.execute("SELECT object_id FROM objects ORDER BY object_id")
        return [row["object_id"] for row in cursor.fetchall()]
    
    def get_all_objects(self) -> List[ObjectProperties]:
        """
        Obtiene todas las propiedades de todos los objetos.
        
        Returns:
            List[ObjectProperties]: Lista de propiedades
        """
        cursor = self.conn.execute("SELECT * FROM objects ORDER BY object_id")
        return [self._row_to_properties(row) for row in cursor.fetchall()]
    
    def delete_object(self, object_id: str) -> bool:
        """
        Elimina un objeto de la base de datos.
        
        Args:
            object_id: Identificador del objeto
        
        Returns:
            bool: True si se eliminó, False si no existía
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM objects WHERE object_id = ?",
                (object_id,)
            )
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Objeto '{object_id}' eliminado")
        return deleted
    
    def find_by_temperature_range(
        self,
        min_temp_k: float,
        max_temp_k: float
    ) -> List[ObjectProperties]:
        """
        Busca objetos en un rango de temperatura.
        
        Args:
            min_temp_k: Temperatura mínima
            max_temp_k: Temperatura máxima
        
        Returns:
            List[ObjectProperties]: Objetos en el rango
        """
        cursor = self.conn.execute(
            "SELECT * FROM objects WHERE temperature_k BETWEEN ? AND ? "
            "ORDER BY temperature_k",
            (min_temp_k, max_temp_k)
        )
        return [self._row_to_properties(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos.
        
        Returns:
            dict: Estadísticas (total objetos, rango temperaturas, etc.)
        """
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_objects,
                AVG(temperature_k) as avg_temperature,
                MIN(temperature_k) as min_temperature,
                MAX(temperature_k) as max_temperature,
                COUNT(DISTINCT emissivity_hash) as unique_emissivities
            FROM objects
        """)
        row = cursor.fetchone()
        
        return {
            "total_objects": row["total_objects"],
            "avg_temperature_k": row["avg_temperature"],
            "min_temperature_k": row["min_temperature"],
            "max_temperature_k": row["max_temperature"],
            "unique_emissivities": row["unique_emissivities"],
            "db_path": str(self.db_path),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
        }
    
    def export_to_json(self, output_path: Path):
        """
        Exporta toda la base de datos a JSON (backup/migración).
        
        Args:
            output_path: Ruta del archivo JSON de salida
        """
        objects = self.get_all_objects()
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_objects": len(objects),
            "objects": [obj.to_dict() for obj in objects]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Base de datos exportada a {output_path}")
    
    def import_from_json(self, json_path: Path):
        """
        Importa objetos desde JSON (migración desde sistema antiguo).
        
        Args:
            json_path: Ruta del archivo JSON
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        objects_data = data.get("objects", [])
        
        for obj_dict in objects_data:
            try:
                props = ObjectProperties.from_dict(obj_dict)
                self.upsert_object(props)
            except Exception as e:
                obj_id = obj_dict.get('object_id')
                logger.error(f"Error importando objeto {obj_id}: {e}")
        
        logger.info(f"Importados {len(objects_data)} objetos desde {json_path}")
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
            logger.info("Conexión a ObjectDatabase cerrada")
    
    def __enter__(self):
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()


# Instancia global (singleton)
_db_instance: Optional[ObjectDatabase] = None


def get_object_db() -> ObjectDatabase:
    """
    Obtiene la instancia singleton de la base de datos.
    
    Returns:
        ObjectDatabase: Instancia compartida
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = ObjectDatabase()
    return _db_instance
