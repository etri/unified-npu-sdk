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
    from unified_sdk.build.api import build_unified_LLM
    from unified_sdk.frontends import resolve_tensorrt_llm_build_request
    from unified_sdk.frontends.types import TensorRTLLMFrontendBuildRequest
    from unified_sdk.options import TensorRTLLMBuildOptions
    from unified_sdk.types import LLMBuildConfig
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)


ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build/fetch a TensorRT-LLM artifact directory. "
            "기본은 model id/local path fetch 이고, custom compile 은 local checkpoint dir 또는 model ref/local model path를 사용합니다."
        )
    )
    parser.add_argument("--model-ref", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--build-mode", choices=("fetch", "custom_compile"), default="fetch")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--model-name", default="tinyllama_trtllm")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    resolved = resolve_tensorrt_llm_build_request(
        TensorRTLLMFrontendBuildRequest(
            model_ref=args.model_ref,
            out_dir=args.out_dir,
            model_name=args.model_name,
            build_mode=args.build_mode,
        )
    )
    cfg = LLMBuildConfig(
        backend="tensorrt",
        model_or_path=args.model_ref,
        out_dir=args.out_dir,
        model_name=args.model_name,
        backend_options=TensorRTLLMBuildOptions(
            build_mode=args.build_mode,
            tokenizer_path=args.tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        ),
        prepared_input=resolved.prepared_input,
    )
    result = build_unified_LLM(cfg)
    print("== TensorRT-LLM build ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"source = {resolved.source_description}")
    print(f"phase = {result.meta_data.get('resolved_phase') if result.meta_data else 'unknown'}")
    print(f"artifact = {result.compiled_model_path}")
    if result.meta_data:
        print(f"artifact_emitted = {result.meta_data.get('artifact_emitted')}")
        print(f"runtime_may_trigger_vendor_build = {result.meta_data.get('runtime_may_trigger_vendor_build')}")
    if result.meta_data:
        print(f"meta = {result.meta_data}")
