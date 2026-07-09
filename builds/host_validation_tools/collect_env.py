from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print(f"[missing command] {command[0]}")
        return

    print(completed.stdout.rstrip())
    if completed.returncode != 0:
        print(f"[command failed] exit_code={completed.returncode}")


def _print_python_env() -> None:
    print("\n== Python Environment ==")
    print("python =", sys.version)
    print("executable =", sys.executable)
    print("platform =", platform.platform())
    print("cwd =", Path.cwd())

    for module_name in ("numpy", "onnx", "torch", "torchvision", "tensorrt", "pycuda"):
        try:
            module = __import__(module_name)
        except Exception as exc:
            print(f"{module_name} import failed: {exc!r}")
            continue
        print(f"{module_name} =", getattr(module, "__version__", "unknown"))

    try:
        import pycuda.driver as cuda

        cuda.init()
        count = cuda.Device.count()
        print("cuda_device_count =", count)
        for i in range(count):
            dev = cuda.Device(i)
            print(f"cuda_device[{i}] = {dev.name()} (cc {dev.compute_capability()})")
    except Exception as exc:
        print(f"pycuda device query failed: {exc!r}")


def main() -> int:
    print("== NVIDIA TensorRT Host Environment ==")
    print("timestamp =", datetime.now().isoformat(timespec="seconds"))

    _run(["uname", "-a"])
    _run(["lsb_release", "-a"])
    _run(["nvidia-smi"])
    _run(["bash", "-lc", "ls -al /dev/nvidia* 2>/dev/null || true"])
    _run(["bash", "-lc", "nvcc --version 2>/dev/null || true"])
    _print_python_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
