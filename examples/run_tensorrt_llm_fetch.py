import argparse
import os
import sys
from pathlib import Path


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

try:
    from unified_sdk.build.api import fetch_unified_LLM
    from unified_sdk.frontends import resolve_tensorrt_llm_fetch_request
    from unified_sdk.frontends.types import TensorRTLLMFrontendFetchRequest
    from unified_sdk.types import LLMFetchConfig
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a TensorRT-LLM runtime fetch contract from a model id or local HF/artifact path."
    )
    parser.add_argument("--model-ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    resolved = resolve_tensorrt_llm_fetch_request(
        TensorRTLLMFrontendFetchRequest(
            model_ref=args.model_ref,
        )
    )
    result = fetch_unified_LLM(
        LLMFetchConfig(
            backend="tensorrt",
            model_ref=args.model_ref,
            prepared_input=resolved.prepared_input,
        )
    )
    print("== TensorRT-LLM fetch ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"source = {resolved.source_description}")
    print(f"phase = {result.meta_data.get('resolved_phase') if result.meta_data else 'unknown'}")
    print(f"model_ref_or_path = {result.model_ref_or_path}")
    if result.meta_data:
        print(f"artifact_emitted = {result.meta_data.get('artifact_emitted')}")
        print(f"runtime_may_trigger_vendor_build = {result.meta_data.get('runtime_may_trigger_vendor_build')}")
        print(f"meta = {result.meta_data}")
