import platform
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
import psutil
import torch


def environment_report():
    gpu = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpu.append(
            {"name": p.name, "vram_bytes": p.total_memory, "compute_capability": [p.major, p.minor]}
        )
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        smi = None
    return {
        "os": platform.platform(),
        "process_id": os.getpid(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_gpus": gpu,
        "nvidia_smi": smi,
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage(".").free,
        "packages": {p: version(p) for p in ["torch", "transformers", "numpy", "pytest"]},
    }
