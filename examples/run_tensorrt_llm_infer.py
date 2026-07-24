import argparse
from pathlib import Path
import sys
import os
import timeit
from typing import Any


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
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM generate through Unified SDK LLM API.")
    parser.add_argument("--engine-path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--prompt", default="What is the capital of South Korea?")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--iters", type=int, default=1)
    return parser


def _render_output(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first
        outputs = getattr(first, "outputs", None)
        if outputs:
            text = getattr(outputs[0], "text", None)
            if isinstance(text, str):
                return text
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
        return str(first)
    return str(result)


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        from unified_sdk.types import LLMRuntimeConfig
        from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM, generate_LLM
    except ImportError:
        print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
        sys.exit(1)

    cfg = LLMRuntimeConfig(
        backend="tensorrt",
        engine_path=args.engine_path,
        tokenizer_path=args.tokenizer_path,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_tokens=args.min_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        extra={
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
        },
    )
    rh = create_runtime_LLM(cfg)

    latencies = []
    result = None
    try:
        for _ in range(args.iters):
            t0 = timeit.default_timer()
            result = generate_LLM(rh, args.prompt)
            t1 = timeit.default_timer()
            latencies.append((t1 - t0) * 1000.0)

        print("== TensorRT-LLM generate ==")
        print(f"repo_root = {REPO_ROOT}")
        print(f"engine = {args.engine_path}")
        print(f"prompt = {args.prompt}")
        print(f"response = {_render_output(result)}")
        print(f"latency_ms = {sum(latencies) / len(latencies):.3f}")
    finally:
        destroy_runtime_LLM(rh)
