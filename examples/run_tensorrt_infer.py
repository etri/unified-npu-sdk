import timeit
from pathlib import Path
import sys

import numpy as np


def _resolve_repo_root() -> Path:
    ws_root = Path("/workspace/unified-sdk")
    if ws_root.is_dir():
        return ws_root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime

if __name__ == "__main__":
    #cfg = RuntimeConfig(
    #    backend="tensorrt",
    #    engine_path="build_output/hrnet_coco_w32_Nx256x192_FP16.engine",
    #    input_name="input.1",
    #    output_name="2901",
    #    input_shape=(1, 3, 256, 192),
    #    use_execute_v3=True,
    #)
    cfg = RuntimeConfig(
        backend="tensorrt",
        engine_path=str(REPO_ROOT / "build_output" / "yolov7_FP32.engine"),
        input_name="images",
        output_name="output",
        input_shape=(1, 3, 640, 640),
        use_execute_v3=True,
    )
    rh = create_runtime(cfg)

    x = np.random.rand(*cfg.input_shape).astype(np.float32)
    for _ in range(1): _ = infer(rh, x)

    iters = 50
    times = []
    for _ in range(iters):
        t0 = timeit.default_timer()
        y = infer(rh, x)
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)

    print(f"Avg latency: {np.mean(times):.3f} ms, shape={y.shape}")
    destroy_runtime(rh)
