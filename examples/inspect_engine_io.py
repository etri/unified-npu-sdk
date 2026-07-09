# examples/inspect_engine_io.py
"""
TensorRT 엔진(.engine)의 입출력 텐서 정보를 출력합니다.
TRT 8/10 을 모두 커버합니다: v3 API(get_tensor_*) 우선, 미지원 시 v2 바인딩 API 사용.
"""
import argparse
from pathlib import Path
import sys
import os

def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    env_root = os.getenv("UNIFIED_SDK_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if _is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    if _is_repo_root(cwd):
        return cwd

    file_root = Path(__file__).resolve().parents[1]
    if _is_repo_root(file_root):
        return file_root

    for candidate in (Path("/workspace/unified-sdk"), Path("/workspace/unified-npu-sdk")):
        if _is_repo_root(candidate):
            return candidate

    return file_root


REPO_ROOT = _resolve_repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the IO tensors of a compiled TensorRT engine.")
    parser.add_argument("engine_path", nargs="?", default=str(REPO_ROOT / "build_output" / "yolov7_FP32.engine"))
    return parser


def inspect(engine_path: str) -> None:
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: 'tensorrt' not found. Run inside the NVIDIA TensorRT container or install it.")
        return

    p = Path(engine_path).expanduser()
    if not p.is_file():
        print(f"Error: file not found - {p}")
        return
    if p.suffix != ".engine":
        print(f"Error: expected a .engine file - {p}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(p.read_bytes())
    if engine is None:
        print(f"Error: failed to deserialize engine - {p}")
        return

    print(f"\n== Engine: {p.name} ==")
    print(f"  path: {p}")
    print(f"  tensorrt: {getattr(trt, '__version__', 'unknown')}")

    has_v3 = hasattr(engine, "get_tensor_name") and hasattr(engine, "get_tensor_mode")
    if has_v3:
        n = engine.num_io_tensors
        print(f"  num_io_tensors: {n}")
        for i in range(n):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)      # trt.TensorIOMode.INPUT / OUTPUT
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)
            print(f"  - {i}: name={name!r}, mode={mode}, dtype={dtype}, shape={tuple(shape)}")
    else:
        nb = engine.num_bindings
        print(f"  num_bindings: {nb}")
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            kind = "INPUT" if is_input else "OUTPUT"
            print(f"  - {i}: name={name!r}, {kind}, dtype={dtype}, shape={tuple(shape)}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    inspect(args.engine_path)
