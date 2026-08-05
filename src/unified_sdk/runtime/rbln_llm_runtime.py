from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from unified_sdk.runtime.registry import register_llm
from unified_sdk.options import resolve_rbln_llm_runtime_options
from unified_sdk.types import LLMRuntimeConfig, LLMRuntimeHandle


_CAPABILITY_FAMILY = "llm.high-level-generation-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "select_runtime_impl",
    "load_vendor_llm",
    "resolve_sampling_params",
    "run_text_generation",
    "extract_text",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "create_runtime_LLM": "vllm.LLM(model=..., tensor_parallel_size=..., max_model_len=..., ...)",
    "sampling": "vllm.SamplingParams(**params)",
    "generate_LLM": "llm.generate(prompts, sampling_params)",
    "destroy_runtime_LLM": "best-effort release of runtime handle",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "vllm.LLM(model=..., tensor_parallel_size=..., max_model_len=..., ...)": "create_runtime_LLM(cfg)",
    "vllm.SamplingParams(**params)": "generate_LLM(rh, prompt, **overrides)",
    "llm.generate(prompts, sampling_params)": "generate_LLM(rh, prompt, **overrides)",
}

_SAMPLING_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "min_tokens")


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime_LLM(cfg)",
            "generate": "generate_LLM(rh, prompt, **overrides)",
            "infer": "infer_LLM(rh, prompt, **overrides)",
            "destroy": "destroy_runtime_LLM(rh)",
        },
        "backend": "rbln",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _extract_text(output: Any) -> str:
    outputs = getattr(output, "outputs", None)
    if outputs:
        text = getattr(outputs[0], "text", None)
        if text is not None:
            return text
    text = getattr(output, "text", None)
    if text is not None:
        return text
    return str(output)


class _RBLNLLMRuntimeAdapter:
    name = "rbln"

    def create(self, cfg: LLMRuntimeConfig) -> LLMRuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"RBLN LLM runtime adapter received backend={cfg.backend!r}")

        options = resolve_rbln_llm_runtime_options(cfg.backend_options, extra=dict(cfg.extra or {}))
        options_meta = options.to_metadata()
        runtime_impl = options.runtime_impl
        if runtime_impl != "vllm":
            raise ValueError(
                "Currently supported RBLN LLM runtime_impl is only 'vllm'. "
                "Use RBLNLLMRuntimeOptions(runtime_impl='vllm') or omit the option."
            )

        try:
            from vllm import LLM
        except Exception as exc:
            raise RuntimeError(
                "vllm-rbln is required for RBLN LLM runtime smoke. "
                "Install vllm-rbln first (see docs.rbln.ai)."
            ) from exc

        engine = str(cfg.engine_path)
        llm_kwargs: Dict[str, Any] = {
            "model": engine,
            "tensor_parallel_size": options.tensor_parallel_size,
            "max_model_len": options.max_model_len,
        }
        llm_kwargs["block_size"] = int(options.block_size) if options.block_size is not None else int(options.max_model_len)
        llm_kwargs["trust_remote_code"] = options.trust_remote_code
        llm_kwargs["enforce_eager"] = options.enforce_eager
        if options.dtype:
            llm_kwargs["dtype"] = options.dtype
        if options.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(options.gpu_memory_utilization)
        if options.additional_config is not None:
            llm_kwargs["additional_config"] = dict(options.additional_config)

        try:
            llm = LLM(**llm_kwargs)
        except Exception as exc:
            message = str(exc)
            if "GatedRepoError" in message or "gated repo" in message.lower() or "401 Client Error" in message:
                raise RuntimeError(
                    "Failed to create RBLN LLM runtime because the selected Hugging Face model is gated or "
                    "requires authentication. For the default public smoke path, try a non-gated model such as "
                    "'Qwen/Qwen3-0.6B', or provide a valid HF_TOKEN if you need a gated model."
                ) from exc
            raise RuntimeError(f"Failed to create RBLN LLM runtime for {engine!r}: {exc}") from exc

        sampling_defaults = {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "min_tokens": cfg.min_tokens,
        }

        return LLMRuntimeHandle(
            backend=self.name,
            engine_path=engine,
            ctx={
                "llm": llm,
                "runtime_impl": runtime_impl,
                "sampling_defaults": sampling_defaults,
                "backend_options": options_meta,
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
                "llm_kwargs": llm_kwargs,
            },
        )

    def generate(self, rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any:
        if not rh.ctx or "llm" not in rh.ctx:
            raise RuntimeError("RBLN LLM RuntimeHandle is closed or invalid")

        llm = rh.ctx["llm"]
        params = dict(rh.ctx.get("sampling_defaults", {}))
        for key, value in overrides.items():
            if key in _SAMPLING_KEYS and value is not None:
                params[key] = value

        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise RuntimeError("vllm-rbln SamplingParams is not available") from exc

        sampling = SamplingParams(**params)
        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)
        if not prompts:
            raise ValueError("prompt must be a non-empty string or list of strings")

        try:
            outputs = llm.generate(prompts, sampling)
        except Exception as exc:
            raise RuntimeError(f"RBLN LLM generate failed: {exc}") from exc

        texts = [_extract_text(item) for item in outputs]
        return texts[0] if single else texts

    def destroy(self, rh: LLMRuntimeHandle) -> None:
        llm = rh.ctx.get("llm") if rh.ctx else None
        if llm is not None:
            for attr in ("llm_engine", "engine"):
                engine = getattr(llm, attr, None)
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        pass
                    break
        rh.ctx.clear()


def create_llm(cfg: LLMRuntimeConfig) -> LLMRuntimeHandle:
    return _RBLNLLMRuntimeAdapter().create(cfg)


def generate_llm(rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    return _RBLNLLMRuntimeAdapter().generate(rh, prompt, **overrides)


def destroy_llm(rh: LLMRuntimeHandle) -> None:
    _RBLNLLMRuntimeAdapter().destroy(rh)


register_llm(_RBLNLLMRuntimeAdapter())
