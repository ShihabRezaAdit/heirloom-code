#!/usr/bin/env python
"""Record the environment (GPU, CUDA, packages, disk) to logs/environment/ and print a summary.

    HEIRLOOM_EXEC_LOCATION=local-pc HEIRLOOM_MACHINE_LABEL=rtx5070-desktop python scripts/detect_env.py
"""
import json
import sys

import _bootstrap  # noqa: F401

from heirloom.utils import envinfo


def main() -> int:
    path = envinfo.write_record()
    rec = json.loads(path.read_text(encoding="utf-8"))
    g = rec["gpu"]
    print(f"saved            : {path}")
    print(f"location / label : {rec['exec_location']} / {rec['machine_label']}")
    print(f"python           : {rec['python']}   RAM {rec['ram_gb']} GB   free disk {rec['repo_free_disk_gb']} GB")
    print(f"torch            : {rec['packages'].get('torch')}  CUDA build {g.get('torch_cuda_build')}  available {g['cuda_available']}")
    for d in g.get("devices", []):
        print(f"gpu {d['index']}            : {d['name']}  {d['total_memory_gb']} GB  sm_{d['capability'].replace('.', '')}")
    if g.get("nvidia_smi"):
        print(f"nvidia-smi       : {g['nvidia_smi']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
