import argparse
import json
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

from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM
from unified_sdk.options import RBLNLLMRuntimeOptions
from unified_sdk.types import LLMRuntimeConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an RBLN LLM model/precompiled artifact.")
    parser.add_argument("model_ref", help="HF model id, local HF path, or precompiled RBLN artifact dir.")
    parser.add_argument("--runtime-impl", choices=("vllm",), default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--load", action="store_true", help="Best-effort: create the actual runtime and print selected kwargs.")
    return parser


def _print_json_if_exists(path: Path, label: str) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"{label}: <failed to parse: {exc}>")
        return
    print(f"{label}:")
    for key in ("model_type", "architectures", "torch_dtype", "max_position_embeddings"):
        if key in data:
            print(f"  {key}: {data[key]}")


if __name__ == "__main__":
    args = _build_parser().parse_args()

    p = Path(args.model_ref).expanduser()
    print("== RBLN LLM inspect ==")
    print("model_ref =", args.model_ref)
    print("runtime_impl =", args.runtime_impl)

    if p.exists():
        print("local_path =", p.resolve())
        if p.is_dir():
            names = sorted(item.name for item in p.iterdir())
            print("dir_entries =", names[:20] + (["..."] if len(names) > 20 else []))
            _print_json_if_exists(p / "config.json", "config.json")
            _print_json_if_exists(p / "generation_config.json", "generation_config.json")
    else:
        print("local_path = <not a local path; treated as model id>")

    if args.load:
        cfg = LLMRuntimeConfig(
            backend="rbln",
            engine_path=args.model_ref,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            backend_options=RBLNLLMRuntimeOptions(runtime_impl=args.runtime_impl),
        )
        rh = create_runtime_LLM(cfg)
        try:
            print("load_ok = True")
            print("llm_kwargs =", rh.ctx.get("llm_kwargs"))
            print("sampling_defaults =", rh.ctx.get("sampling_defaults"))
        finally:
            destroy_runtime_LLM(rh)
