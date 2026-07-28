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
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM, generate_LLM
from unified_sdk.types import LLMRuntimeConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with an RBLN LLM via Unified SDK.")
    parser.add_argument("--engine-path", required=True, help="HF model id, local HF path, or precompiled RBLN artifact dir.")
    parser.add_argument("--prompt", default="What is the capital of South Korea?")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--runtime-impl", choices=("vllm",), default="vllm")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    cfg = LLMRuntimeConfig(
        backend="rbln",
        engine_path=args.engine_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_tokens=args.min_tokens,
        extra={
            key: value
            for key, value in {
                "runtime_impl": args.runtime_impl,
                "block_size": args.block_size,
                "trust_remote_code": args.trust_remote_code if args.trust_remote_code else None,
            }.items()
            if value is not None
        },
    )

    rh = create_runtime_LLM(cfg)
    try:
        t0 = timeit.default_timer()
        text = generate_LLM(rh, args.prompt)
        dt = (timeit.default_timer() - t0) * 1000
        print("== RBLN LLM generate ==")
        print("engine =", args.engine_path)
        print("runtime_impl =", args.runtime_impl)
        print("prompt =", args.prompt)
        print("---- output ----")
        print(text)
        print("----------------")
        print(f"latency_ms = {dt:.1f}")
    finally:
        destroy_runtime_LLM(rh)
