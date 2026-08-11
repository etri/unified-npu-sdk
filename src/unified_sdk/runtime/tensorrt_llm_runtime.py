from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.options import resolve_tensorrt_llm_runtime_options
from unified_sdk.runtime.registry import register_llm
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


def _looks_like_local_path(model_ref: str) -> bool:
    p = Path(model_ref).expanduser()
    return p.is_absolute() or model_ref.startswith("./") or model_ref.startswith("../") or model_ref.startswith(("artifacts/", "build_output/", "models/"))


def _detect_runtime_entry_kind(model_ref: str) -> str:
    p = Path(model_ref).expanduser()
    if p.exists():
        if p.is_dir():
            markers = ("config.json", "executor_config.json", "engine_config.json")
            if any((p / marker).exists() for marker in markers) or any(p.glob("*.engine")):
                return "local_artifact_dir"
        return "local_model_path"
    return "model_id"


def _normalize_llm_kwargs(cfg: LLMRuntimeConfig, options, model_ref: str) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": model_ref,
        "tensor_parallel_size": options.tensor_parallel_size,
        "max_seq_len": options.max_model_len,
        "tokenizer": str(options.tokenizer_path) if options.tokenizer_path else model_ref,
    }
    if options.dtype:
        llm_kwargs["dtype"] = options.dtype
    if options.trust_remote_code is not None:
        llm_kwargs["trust_remote_code"] = bool(options.trust_remote_code)
    return llm_kwargs


class _TensorRTLLMRuntimeAdapter:
    name = "tensorrt"

    def create(self, cfg: LLMRuntimeConfig) -> LLMRuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT-LLM runtime adapter received backend={cfg.backend!r}")

        _ensure_positive_int(cfg.max_tokens, "LLMRuntimeConfig.max_tokens")
        options = resolve_tensorrt_llm_runtime_options(cfg.backend_options)
        _ensure_positive_int(options.max_model_len, "TensorRTLLMRuntimeOptions.max_model_len")
        _ensure_positive_int(options.tensor_parallel_size, "TensorRTLLMRuntimeOptions.tensor_parallel_size")

        try:
            from tensorrt_llm import LLM
        except Exception as exc:
            raise RuntimeError(
                "tensorrt_llm is required for TensorRT-LLM runtime/generation. Install it in the container or host env first."
            ) from exc

        model_ref = str(cfg.model_ref_or_path)
        runtime_entry_kind = _detect_runtime_entry_kind(model_ref)
        runtime_mode = "artifact_runtime" if runtime_entry_kind == "local_artifact_dir" else "convenience_model_ref_runtime"
        runtime_may_trigger_vendor_build = runtime_entry_kind != "local_artifact_dir"
        if _looks_like_local_path(model_ref):
            local_ref = Path(model_ref).expanduser()
            if not local_ref.exists():
                raise FileNotFoundError(
                    "engine_path was interpreted as a local TensorRT-LLM artifact/model directory, "
                    f"but it does not exist: {local_ref}. If you intended a Hugging Face repo id, pass an explicit repo id."
                )

        llm_kwargs = _normalize_llm_kwargs(cfg, options, model_ref)
        try:
            llm = LLM(**llm_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to create TensorRT-LLM runtime for {model_ref}: {exc}") from exc

        return LLMRuntimeHandle(
            backend=self.name,
            model_ref_or_path=model_ref,
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
                "runtime_options": options.to_metadata(),
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
                "runtime_entry_kind": runtime_entry_kind,
                "runtime_mode": runtime_mode,
                "runtime_may_trigger_vendor_build": runtime_may_trigger_vendor_build,
            },
        )

    def infer(self, rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any:
        if not rh.ctx or "llm" not in rh.ctx:
            raise RuntimeError("TensorRT-LLM RuntimeHandle is closed or invalid")
        try:
            from tensorrt_llm import SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "tensorrt_llm is required for TensorRT-LLM generation. Install it in the container or host env first."
            ) from exc
        llm = rh.ctx["llm"]
        sampling_dict = dict(rh.ctx.get("sampling_defaults", {}))
        sampling_dict.update({k: v for k, v in overrides.items() if v is not None})
        try:
            sampling = SamplingParams(**sampling_dict)
            return llm.generate(prompt, sampling_params=sampling)
        except Exception as exc:
            raise RuntimeError(f"TensorRT-LLM generate failed for {rh.model_ref_or_path}: {exc}") from exc

    def destroy(self, rh: LLMRuntimeHandle) -> None:
        llm = (rh.ctx or {}).get("llm")
        if llm is not None:
            _best_effort_close(llm)
        (rh.ctx or {}).clear()


register_llm(_TensorRTLLMRuntimeAdapter())
