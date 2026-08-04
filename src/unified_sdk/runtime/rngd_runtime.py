from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from unified_sdk.options import RNGDRuntimeOptions
from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle


_CAPABILITY_FAMILY = "llm.artifact-and-generation-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "load_llm_or_local_model",
    "optionally_attach_fxb",
    "resolve_sampling_params",
    "run_text_generation",
    "extract_text",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "create_runtime_LLM": "furiosa_llm.LLM(model_id_or_path, fxb=..., devices=...)",
    "sampling": "furiosa_llm.SamplingParams(**params)",
    "generate_LLM": "llm.generate(prompts, sampling)",
    "destroy_runtime_LLM": "llm.shutdown/close/dispose best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "furiosa_llm.LLM(model_id_or_path, fxb=..., devices=...)": "create_runtime_LLM(cfg)",
    "furiosa_llm.SamplingParams(**params)": "infer_LLM(rh, prompt, **overrides) / generate_LLM(...)",
    "llm.generate(prompts, sampling)": "infer_LLM(rh, prompt, **overrides) / generate_LLM(...)",
    "RequestOutput.outputs[0].text": "infer_LLM(...) return str or list[str]",
    "llm.shutdown/close/dispose": "destroy_runtime_LLM(rh)",
}


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime_LLM(cfg)",
            "infer": "infer_LLM(rh, prompt, **overrides)",
            "generate": "generate_LLM(rh, prompt, **overrides)",
            "destroy": "destroy_runtime_LLM(rh)",
        },
        "backend": "rngd",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


_SAMPLING_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "min_tokens")


def _extract_text(request_output: Any) -> str:
    """furiosa_llm generate() 결과의 단일 RequestOutput 에서 텍스트를 뽑는다."""
    outputs = getattr(request_output, "outputs", None)
    if outputs:
        text = getattr(outputs[0], "text", None)
        if text is not None:
            return text
    # best-effort fallback
    text = getattr(request_output, "text", None)
    if text is not None:
        return text
    return str(request_output)


class _RNGDRuntime:
    """FuriosaAI RNGD runtime adapter — wraps furiosa_llm.LLM.

    참조 API (developer.furiosa.ai):
      - from furiosa_llm import LLM, SamplingParams
      - llm = LLM(model_id_or_path, fxb=optional_fxb_path, devices=...)
      - sp = SamplingParams(max_tokens=..., temperature=..., top_p=..., top_k=..., min_tokens=...)
      - outputs = llm.generate([prompt], sp)  -> outputs[i].outputs[0].text
    """

    name = "rngd"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"RNGD runtime adapter received backend={cfg.backend!r}")

        engine = str(cfg.engine_path)
        try:
            from furiosa_llm import LLM
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(
                "furiosa-llm is required to serve an RNGD model. "
                "Install furiosa-llm first (see developer.furiosa.ai)."
            ) from exc

        runtime_options = RNGDRuntimeOptions.from_raw(
            cfg.backend_options,
            legacy_fxb_path=cfg.fxb_path,
            legacy_devices=cfg.devices,
        )
        fxb_path = str(runtime_options.fxb_path) if runtime_options.fxb_path else None
        llm_kwargs: Dict[str, Any] = {}
        if runtime_options.devices:
            llm_kwargs["devices"] = runtime_options.devices
        if fxb_path:
            llm_kwargs["fxb"] = fxb_path

        try:
            llm = LLM(engine, **llm_kwargs)
            if fxb_path:
                source = "model_or_path+fxb"
            elif Path(engine).is_dir():
                source = "local_model_path"
            else:
                source = "model_id"
        except Exception as exc:
            raise RuntimeError(f"Failed to load RNGD LLM for {engine!r}: {exc}") from exc

        sampling_defaults = {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "min_tokens": cfg.min_tokens,
        }

        return RuntimeHandle(
            backend=self.name,
            engine_path=engine,
            ctx={
                "llm": llm,
                "source": source,
                "devices": runtime_options.devices,
                "fxb_path": fxb_path,
                "sampling_defaults": sampling_defaults,
                "backend_options": runtime_options.to_metadata(),
                "extra": dict(cfg.extra or {}),
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def _make_sampling_params(self, rh: RuntimeHandle, overrides: Dict[str, Any]):
        from furiosa_llm import SamplingParams

        params = dict(rh.ctx.get("sampling_defaults", {}))
        for key, value in overrides.items():
            if key in _SAMPLING_KEYS and value is not None:
                params[key] = value
        return SamplingParams(**params)

    def infer(self, rh: RuntimeHandle, prompt: Union[str, List[str]], **overrides: Any) -> Union[str, List[str]]:
        return self.generate(rh, prompt, **overrides)

    def generate(self, rh: RuntimeHandle, prompt: Union[str, List[str]], **overrides: Any) -> Union[str, List[str]]:
        if not rh.ctx or "llm" not in rh.ctx:
            raise RuntimeError("RNGD RuntimeHandle is closed or invalid")

        llm = rh.ctx["llm"]
        sampling = self._make_sampling_params(rh, overrides)

        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)
        if not prompts:
            raise ValueError("prompt must be a non-empty string or list of strings")

        try:
            outputs = llm.generate(prompts, sampling)
        except Exception as exc:
            raise RuntimeError(f"RNGD generate failed: {exc}") from exc

        texts = [_extract_text(o) for o in outputs]
        return texts[0] if single else texts

    def destroy(self, rh: RuntimeHandle) -> None:
        llm = rh.ctx.get("llm") if rh.ctx else None
        if llm is not None:
            for method in ("shutdown", "close", "dispose"):
                fn = getattr(llm, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
        rh.ctx.clear()


register(_RNGDRuntime())
