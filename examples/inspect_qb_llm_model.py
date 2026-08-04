"""
Inspect a QB transformer/LLM MXQ with qbruntime-specific cache metadata.
This script is intended for low-level LLM smoke on precompiled Mobilint MXQ models.
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
DEFAULT_MODEL = REPO_ROOT / "models" / "Llama-3.2-1B-Instruct.mxq"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect cache-aware metadata of a QB LLM/transformer MXQ.")
    parser.add_argument("model_path", nargs="?", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--core-mode",
        default=os.getenv("MBLT_CORE_MODE", "global8"),
        help="LLM/transformer MXQ load core mode. Multi-core-mode MXQ는 auto 대신 explicit mode가 필요할 수 있습니다.",
    )
    return parser


def _safe_call(obj, name: str):
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            return fn()
        except Exception as exc:
            return f"<failed: {type(exc).__name__}: {exc}>"
    return "<unavailable>"


if __name__ == "__main__":
    args = _build_parser().parse_args()
    p = Path(args.model_path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"Error: file not found - {p}")
    if p.suffix != ".mxq":
        raise SystemExit(f"Error: expected a .mxq file - {p}")

    try:
        from unified_sdk.options import QBSequenceRuntimeOptions
        from unified_sdk.sequence_runtime import create_sequence_runtime, destroy_sequence_runtime
        from unified_sdk.sequence_runtime.types import SequenceRuntimeConfig
        from qbruntime import type as qb_type
    except Exception as exc:
        raise SystemExit(f"Error: unified_sdk runtime and qbruntime are required ({type(exc).__name__}: {exc})")

    cfg = SequenceRuntimeConfig(
        backend="qb",
        engine_path=str(p),
        input_name="input",
        output_name="output",
        input_shape=(1,),
        backend_options=QBSequenceRuntimeOptions(
            core_mode=args.core_mode,
            allow_dynamic_shape=True,
        ),
    )
    rh = create_sequence_runtime(cfg)
    model = rh.ctx["model"]
    try:
        print("== QB LLM model inspect ==")
        print(f"path = {p}")
        print("core_mode_arg =", args.core_mode)
        print("available_devices =", _safe_call(qb_type, "get_available_device_numbers"))
        print("core_mode =", _safe_call(model, "get_core_mode"))
        print("target_cores =", _safe_call(model, "get_target_cores"))
        print("input_shapes =", _safe_call(model, "get_model_input_shape"))
        print("output_shapes =", _safe_call(model, "get_model_output_shape"))
        print("input_dtype =", _safe_call(model, "get_model_input_data_type"))
        print("output_dtype =", _safe_call(model, "get_model_output_data_type"))
        print("input_buffer_info =", _safe_call(model, "get_input_buffer_info"))
        print("output_buffer_info =", _safe_call(model, "get_output_buffer_info"))
        print("cache_infos =", _safe_call(model, "get_cache_infos"))
    finally:
        destroy_sequence_runtime(rh)
