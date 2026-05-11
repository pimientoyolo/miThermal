import os
import time
import subprocess
import csv
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Utility to monitor and record GPU/CPU performance during rendering.
    Records to backend/output/performance_stats.csv
    """
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            from src.config import OUTPUT_DIR
            self.output_path = Path(OUTPUT_DIR) / "performance_stats.csv"
        else:
            self.output_path = Path(output_dir) / "performance_stats.csv"
            
        self.header = [
            "timestamp", "render_time_s", "width", "height", "spp", "num_bands",
            "gpu_model", "gpu_load_pct", "vram_used_mb", "vram_total_mb", 
            "ram_used_gb", "ram_total_gb", "cpu_load_pct"
        ]
        
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Creates the CSV file with header if it doesn't exist."""
        if not self.output_path.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def _get_gpu_stats(self):
        """Gets GPU stats using nvidia-smi."""
        try:
            cmd = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            # If multiple GPUs, take the first one
            gpu_info = output.split('\n')[0].split(', ')
            return {
                "model": gpu_info[0],
                "load": float(gpu_info[1]),
                "vram_used": float(gpu_info[2]),
                "vram_total": float(gpu_info[3])
            }
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {e}")
            return {"model": "N/A", "load": 0, "vram_used": 0, "vram_total": 0}

    def _get_system_stats(self):
        """Gets RAM and CPU stats."""
        try:
            import psutil
            ram = psutil.virtual_memory()
            return {
                "ram_used": ram.used / (1024**3),
                "ram_total": ram.total / (1024**3),
                "cpu_load": psutil.cpu_percent()
            }
        except ImportError:
            # Fallback for RAM using /proc/meminfo on Linux
            try:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                total = int(lines[0].split()[1]) / (1024**2) # GB
                free = int(lines[1].split()[1]) / (1024**2) # GB
                return {"ram_used": total - free, "ram_total": total, "cpu_load": 0}
            except Exception:
                return {"ram_used": 0, "ram_total": 0, "cpu_load": 0}

    def record_render_event(self, start_time, config):
        """
        Records a rendering event with its performance metrics.
        
        Args:
            start_time: time.time() value when render started
            config: dict containing camera and scene parameters
        """
        end_time = time.time()
        duration = end_time - start_time
        
        gpu = self._get_gpu_stats()
        sys = self._get_system_stats()
        
        camera = config.get("camera", {})
        
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{duration:.4f}",
            camera.get("width", 0),
            camera.get("height", 0),
            camera.get("spp", 0),
            config.get("num_bands", 0),
            gpu["model"],
            gpu["load"],
            gpu["vram_used"],
            gpu["vram_total"],
            f"{sys['ram_used']:.2f}",
            f"{sys['ram_total']:.2f}",
            sys["cpu_load"]
        ]
        
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        logger.info(f"Performance stats recorded to {self.output_path}")
