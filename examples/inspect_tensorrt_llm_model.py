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
    parser = argparse.ArgumentParser(description="Inspect a TensorRT-LLM model ref or prebuilt artifact dir.")
    parser.add_argument("engine_path", nargs="?", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load", action="store_true", help="Best-effort runtime creation까지 확인합니다.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        from unified_sdk.types import LLMRuntimeConfig
        from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM
    except ImportError:
        print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
        sys.exit(1)

    p = Path(args.engine_path).expanduser()
    print("== TensorRT-LLM inspect ==")
    print(f"path_arg = {args.engine_path}")
    print(f"resolved_exists = {p.exists()}")
    print(f"is_dir = {p.is_dir()}")
    print(f"tokenizer_path = {args.tokenizer_path}")
    if p.is_dir():
        entries = sorted(child.name for child in p.iterdir())
        print(f"dir_entries = {entries}")

    if args.load:
        cfg = LLMRuntimeConfig(
            backend="tensorrt",
            engine_path=args.engine_path,
            tokenizer_path=args.tokenizer_path,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            extra={
                "dtype": args.dtype,
                "trust_remote_code": args.trust_remote_code,
            },
        )
        rh = create_runtime_LLM(cfg)
        try:
            print("load_ok = True")
            print(f"llm_kwargs = {rh.ctx.get('llm_kwargs')}")
        finally:
            destroy_runtime_LLM(rh)
