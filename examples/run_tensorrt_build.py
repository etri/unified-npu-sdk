from pathlib import Path
import sys


def _resolve_repo_root() -> Path:
    ws_root = Path("/workspace/unified-sdk")
    if ws_root.is_dir():
        return ws_root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    from unified_sdk.types import BuildConfig
    from unified_sdk.build.api import build_unified
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)


if __name__ == "__main__":
    #cfg = BuildConfig(
    #   backend="tensorrt",
    #   model_or_path="models/hrnet_coco_w32_Nx256x192.onnx",
    #   out_dir="build_output",
    #   model_name="hrnet_coco_w32_Nx256x192",
    #   precision="fp16",
    #   input_name="input.1",
    #   min_input_shape=(1, 3, 256, 192),
    #   opt_input_shape=(4, 3, 256, 192),
    #   max_input_shape=(30, 3, 256, 192),
    #)
    model_path = REPO_ROOT / "models" / "yolov7.onnx"
    out_dir = REPO_ROOT / "build_output"

    cfg = BuildConfig(
        backend="tensorrt",
        model_or_path=str(model_path),
        out_dir=str(out_dir),
        model_name="yolov7",
        precision="fp32",
        input_name="images",
        min_input_shape=(1, 3, 640, 640),
        opt_input_shape=(1, 3, 640, 640),
        max_input_shape=(1, 3, 640, 640),
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
