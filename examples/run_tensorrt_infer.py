# examples/run_tensorrt_infer.py
import argparse
import timeit
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


ENGINE_PATH = REPO_ROOT / "build_output" / "yolov7_FP32.engine"


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = value.replace("x", ",").split(",")
    try:
        shape = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {value!r}") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape must contain positive integers: {value!r}")
    return shape


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a compiled TensorRT .engine.")
    parser.add_argument("--engine-path", type=Path, default=ENGINE_PATH)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 640, 640))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-execute-v3", action="store_true", help="execute_v2(bindings) 경로 강제.")
    parser.add_argument("--allow-dynamic-shape", action="store_true")
    return parser


def _check_files(engine_path: Path):
    if not engine_path.is_file():
        raise FileNotFoundError(f"필요한 파일이 없습니다:\n- engine: {engine_path}")


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
    except ImportError:
        print("Error: 'numpy' is required for the TensorRT inference example.")
        sys.exit(1)

    from unified_sdk.types import RuntimeConfig
    from unified_sdk.runtime import create_runtime, infer, destroy_runtime

    engine_path = args.engine_path.expanduser().resolve()
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    _check_files(engine_path)

    cfg = RuntimeConfig(
        backend="tensorrt",
        engine_path=str(engine_path),
        input_name=args.input_name,
        output_name=args.output_name,
        input_shape=args.input_shape,
        use_execute_v3=not args.no_execute_v3,
        extra={"allow_dynamic_shape": args.allow_dynamic_shape},
    )

    rh = create_runtime(cfg)

    rng = np.random.default_rng(args.seed)
    x = rng.random(args.input_shape, dtype=np.float32)

    for _ in range(args.warmup):
        _ = infer(rh, x)

    times = []
    y = None
    for _ in range(args.iters):
        t0 = timeit.default_timer()
        y = infer(rh, x)
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)

    print("== TensorRT Inference ==")
    print(f"engine = {engine_path}")
    print(f"input  = {args.input_name} {args.input_shape}")
    print(f"output = {args.output_name} {y.shape}")
    if y.ndim >= 2 and y.shape[0] >= 1:
        flat = y.reshape(y.shape[0], -1)
        print(f"argmax(batch0) = {int(np.argmax(flat[0]))}")
    print(f"latency_ms_avg = {np.mean(times):.3f}")
    print(f"latency_ms_min = {np.min(times):.3f}")
    print(f"latency_ms_max = {np.max(times):.3f}")
    print(f"(execute_v3={cfg.use_execute_v3}, iters={args.iters})")

    destroy_runtime(rh)
