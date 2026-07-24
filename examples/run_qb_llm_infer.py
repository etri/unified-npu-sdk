import argparse
from pathlib import Path
import sys
import os
import timeit


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
DEFAULT_ENGINE = REPO_ROOT / "models" / "Llama-3.2-1B-Instruct.mxq"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Low-level QB transformer/LLM runtime smoke. "
            "This is a cache-aware infer smoke, not a high-level text generate API."
        )
    )
    parser.add_argument("--engine-path", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--core-mode",
        default=os.getenv("MBLT_CORE_MODE", "global8"),
        help="LLM/transformer MXQ load core mode. Multi-core-mode MXQ는 auto 대신 explicit mode가 필요할 수 있습니다.",
    )
    parser.add_argument("--cache-size", type=int, default=0, help="KV cache token count for a single-step infer smoke.")
    parser.add_argument(
        "--batch-seq-lens",
        default=None,
        help="Batch LLM smoke. 예: 10,80 -> BatchParam(10,0,0), BatchParam(80,0,1)",
    )
    return parser


def _parse_batch_seq_lens(value: str | None) -> list[int]:
    if not value:
        return []
    seq_lens = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seq_lens or any(v <= 0 for v in seq_lens):
        raise ValueError("--batch-seq-lens must contain positive integers")
    return seq_lens


def _dtype_to_numpy(data_type, np):
    name = str(data_type).lower()
    if "uint8" in name:
        return np.uint8
    if "int8" in name:
        return np.int8
    if "float16" in name:
        return np.float16
    return np.float32


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
        from unified_sdk.runtime import create_runtime, destroy_runtime, infer
        from unified_sdk.types import BatchParam, RuntimeConfig
    except Exception as exc:
        raise SystemExit(f"Error: unified_sdk runtime and numpy are required ({type(exc).__name__}: {exc})")

    engine_path = args.engine_path.expanduser().resolve()
    if not engine_path.is_file():
        raise SystemExit(f"Error: file not found - {engine_path}")

    cfg = RuntimeConfig(
        backend="qb",
        engine_path=str(engine_path),
        input_name="input",
        output_name="output",
        input_shape=(1,),
        extra={"core_mode": args.core_mode, "allow_dynamic_shape": True},
    )
    rh = create_runtime(cfg)
    model = rh.ctx["model"]
    try:
        input_shapes = model.get_model_input_shape()
        input_dtype = model.get_model_input_data_type()
        np_dtype = _dtype_to_numpy(input_dtype, np)
        seq_lens = _parse_batch_seq_lens(args.batch_seq_lens)

        if not input_shapes:
            raise RuntimeError("Model did not report any input shapes")
        shape = list(input_shapes[0])
        if seq_lens:
            if len(shape) < 2:
                raise RuntimeError(
                    "Batch LLM smoke expects an input shape with at least 2 dimensions "
                    "(e.g. (1, seq_len, hidden_dim))"
                )
            shape[1] = sum(seq_lens)
        x = np.zeros(shape, dtype=np_dtype)

        params = None
        if seq_lens:
            params = [BatchParam(sequence_length=seq_len, cache_size=args.cache_size, cache_id=idx) for idx, seq_len in enumerate(seq_lens)]

        # warmup
        _ = infer(rh, x, cache_size=args.cache_size, batch_params=params)

        times = []
        outputs = None
        for _ in range(args.iters):
            t0 = timeit.default_timer()
            outputs = infer(rh, x, cache_size=args.cache_size, batch_params=params)
            times.append((timeit.default_timer() - t0) * 1000)

        if isinstance(outputs, list):
            output_shapes = [tuple(getattr(out, "shape", ())) for out in outputs]
        else:
            output_shapes = [tuple(getattr(outputs, "shape", ()))]
        print("== QB LLM infer smoke ==")
        print("engine =", engine_path)
        print("runtime_api =", "infer(rh, input_array, cache_size=..., batch_params=...)")
        print("core_mode =", args.core_mode)
        print("input_dtype =", input_dtype)
        print("input_shape =", tuple(shape))
        print("cache_size =", args.cache_size)
        print("batch_seq_lens =", seq_lens or None)
        print("output_shapes =", output_shapes)
        print(f"avg_latency_ms = {sum(times) / len(times):.3f}")
    finally:
        destroy_runtime(rh)
