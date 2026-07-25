from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.types import LLMRuntimeConfig, LLMRuntimeHandle


_CAPABILITY_FAMILY = "llm.high-level-generate-runtime"
_RUNTIME_PIPELINE = (
    "validate_llm_runtime_config",
    "instantiate_tensorrt_llm",
    "prepare_sampling_params",
    "run_generate",
    "best_effort_shutdown",
)
_VENDOR_API_MAP = {
    "create": "tensorrt_llm.LLM(model=..., tokenizer=..., ...)",
    "sampling": "tensorrt_llm.SamplingParams(...)",
    "generate": "llm.generate(prompts, sampling_params=...)",
    "destroy": "llm.shutdown()/close()/dispose() best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "tensorrt_llm.LLM(model=..., tokenizer=..., ...)": "create_runtime_LLM(cfg)",
    "tensorrt_llm.SamplingParams(...)": "generate_LLM(rh, prompt, **overrides)",
    "llm.generate(prompts, sampling_params=...)": "generate_LLM(rh, prompt, **overrides)",
    "llm.shutdown()/close()/dispose()": "destroy_runtime_LLM(rh)",
}


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime_LLM(cfg)",
            "generate": "generate_LLM(rh, prompt, **overrides)",
            "destroy": "destroy_runtime_LLM(rh)",
        },
        "backend": "tensorrt",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _ensure_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _best_effort_close(obj: Any) -> None:
    for name in ("shutdown", "close", "dispose"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _normalize_llm_kwargs(cfg: LLMRuntimeConfig, extra: Dict[str, Any], model_ref: str, tokenizer_path: str | None) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": model_ref,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        # TensorRT-LLM 1.x torch backend uses max_seq_len instead of max_model_len.
        "max_seq_len": cfg.max_model_len,
    }
    llm_kwargs["tokenizer"] = tokenizer_path or model_ref
    if extra.get("dtype"):
        llm_kwargs["dtype"] = extra["dtype"]
    if extra.get("trust_remote_code") is not None:
        llm_kwargs["trust_remote_code"] = bool(extra["trust_remote_code"])
    return llm_kwargs


def create_llm(cfg: LLMRuntimeConfig) -> LLMRuntimeHandle:
    if cfg.backend != "tensorrt":
        raise ValueError(f"TensorRT-LLM runtime adapter received backend={cfg.backend!r}")

    _ensure_positive_int(cfg.max_model_len, "LLMRuntimeConfig.max_model_len")
    _ensure_positive_int(cfg.max_tokens, "LLMRuntimeConfig.max_tokens")
    _ensure_positive_int(cfg.tensor_parallel_size, "LLMRuntimeConfig.tensor_parallel_size")

    try:
        from tensorrt_llm import LLM
    except Exception as exc:
        raise RuntimeError(
            "tensorrt_llm is required for TensorRT-LLM runtime/generation. "
            "Install it in the container or host env first."
        ) from exc

    extra = dict(cfg.extra or {})
    model_ref = str(cfg.engine_path)
    tokenizer_path = str(cfg.tokenizer_path) if cfg.tokenizer_path else None

    llm_kwargs = _normalize_llm_kwargs(cfg, extra, model_ref, tokenizer_path)

    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to create TensorRT-LLM runtime for {model_ref}: {exc}") from exc

    return LLMRuntimeHandle(
        backend="tensorrt",
        engine_path=model_ref,
        ctx={
            "llm": llm,
            "llm_kwargs": llm_kwargs,
            "sampling_defaults": {
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "min_tokens": cfg.min_tokens,
            },
            "capability_family": _CAPABILITY_FAMILY,
            "runtime_pipeline": _RUNTIME_PIPELINE,
            "vendor_api_map": _VENDOR_API_MAP,
        },
    )


def generate_llm(rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    if not rh.ctx or "llm" not in rh.ctx:
        raise RuntimeError("TensorRT-LLM RuntimeHandle is closed or invalid")

    try:
        from tensorrt_llm import SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "tensorrt_llm is required for TensorRT-LLM generation. "
            "Install it in the container or host env first."
        ) from exc

    llm = rh.ctx["llm"]
    sampling_dict = dict(rh.ctx.get("sampling_defaults", {}))
    sampling_dict.update({k: v for k, v in overrides.items() if v is not None})

    try:
        sampling = SamplingParams(**sampling_dict)
        return llm.generate(prompt, sampling_params=sampling)
    except Exception as exc:
        raise RuntimeError(f"TensorRT-LLM generate failed for {rh.engine_path}: {exc}") from exc


def destroy_llm(rh: LLMRuntimeHandle) -> None:
    llm = (rh.ctx or {}).get("llm")
    if llm is not None:
        _best_effort_close(llm)
    (rh.ctx or {}).clear()
