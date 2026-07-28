from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.types import BuildResult, LLMBuildConfig


_CAPABILITY_FAMILY = "llm.high-level-generate-builder"
_BUILD_PIPELINE = (
    "validate_llm_build_config",
    "resolve_build_mode",
    "optional_hf_or_local_model_fetch",
    "optional_tensorrt_llm_compile",
    "optional_artifact_save",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "fetch": "model ref / local model path passthrough",
    "compile": "tensorrt_llm.LLM(model=..., ...)",
    "save_artifact": "llm.save(engine_dir) [if supported by installed TensorRT-LLM]",
    "artifact": "TensorRT-LLM engine dir",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "model ref / local model path passthrough": "build_unified_LLM(cfg) with build_mode=fetch",
    "tensorrt_llm.LLM(model=..., ...)": "build_unified_LLM(cfg)",
    "llm.save(engine_dir)": "build_unified_LLM(cfg) with build_mode=llm_api_compile",
    "TensorRT-LLM engine dir": "BuildResult.compiled_model_path",
}


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified_LLM(cfg)",
        "backend": "tensorrt",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _ensure_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"LLMBuildConfig.{field_name} must be a positive integer")
    return value


def _best_effort_close(obj: Any) -> None:
    for name in ("shutdown", "close", "dispose"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _normalize_llm_kwargs(cfg: LLMBuildConfig, extra: Dict[str, Any], model_ref: str) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": model_ref,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        # TensorRT-LLM 1.x torch backend uses max_seq_len instead of max_model_len.
        "max_seq_len": cfg.max_model_len,
    }
    if extra.get("tokenizer_path"):
        llm_kwargs["tokenizer"] = str(extra["tokenizer_path"])
    if extra.get("dtype"):
        llm_kwargs["dtype"] = extra["dtype"]
    if extra.get("trust_remote_code") is not None:
        llm_kwargs["trust_remote_code"] = bool(extra["trust_remote_code"])
    return llm_kwargs


def build_llm(cfg: LLMBuildConfig) -> BuildResult:
    if cfg.backend != "tensorrt":
        raise ValueError(f"TensorRT-LLM build adapter received backend={cfg.backend!r}")

    extra = dict(cfg.extra or {})
    build_mode = str(extra.get("build_mode", "fetch")).strip().lower()
    model_ref = str(cfg.model_or_path)

    _ensure_positive_int(cfg.max_model_len, "max_model_len")
    _ensure_positive_int(cfg.tensor_parallel_size, "tensor_parallel_size")

    if build_mode == "fetch":
        return BuildResult(
            backend="tensorrt",
            compiled_model_path=model_ref,
            meta_data={
                "backend": "tensorrt",
                "track": "llm",
                "build_mode": "fetch",
                "model_ref": model_ref,
                "extra": extra,
                "capability_family": _CAPABILITY_FAMILY,
                "build_pipeline": _BUILD_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    if build_mode != "llm_api_compile":
        raise ValueError("LLM build_mode must be one of: fetch, llm_api_compile")

    try:
        from tensorrt_llm import LLM
    except Exception as exc:
        raise RuntimeError(
            "tensorrt_llm is required for TensorRT-LLM compile/build. "
            "Install it in the container or host env first."
        ) from exc

    if not hasattr(LLM, "save"):
        raise RuntimeError(
            "TensorRT-LLM build_mode=llm_api_compile is currently unsupported in this trt-only llm flavor. "
            "The installed official TensorRT-LLM release container exposes the PyTorch backend LLM API, "
            "and this LLM class does not provide save(engine_dir). "
            "Use build_mode=fetch for model-id generation, or provide an already prepared local artifact dir for 7-b."
        )

    compiled_dir = Path(cfg.out_dir) / cfg.model_name
    compiled_dir.parent.mkdir(parents=True, exist_ok=True)

    llm_kwargs = _normalize_llm_kwargs(cfg, extra, model_ref)

    llm = None
    try:
        llm = LLM(**llm_kwargs)
        llm.save(str(compiled_dir))
    except Exception as exc:
        raise RuntimeError(f"TensorRT-LLM compile/save failed for {model_ref}: {exc}") from exc
    finally:
        if llm is not None:
            _best_effort_close(llm)

    return BuildResult(
        backend="tensorrt",
        compiled_model_path=str(compiled_dir),
        meta_data={
            "backend": "tensorrt",
            "track": "llm",
            "build_mode": build_mode,
            "model_ref": model_ref,
            "compiled_dir": str(compiled_dir),
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "max_model_len": cfg.max_model_len,
            "llm_kwargs": llm_kwargs,
            "extra": extra,
            "capability_family": _CAPABILITY_FAMILY,
            "build_pipeline": _BUILD_PIPELINE,
            "vendor_api_map": _VENDOR_API_MAP,
        },
    )
