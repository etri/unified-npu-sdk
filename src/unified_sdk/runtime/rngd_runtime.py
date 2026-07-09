from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle


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

    참조 API (developer.furiosa.ai / 사내 가이드 code-21):
      - from furiosa_llm import LLM, SamplingParams
      - llm = LLM(model_id)  또는  LLM.from_artifacts(artifact_dir)
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

        try:
            if Path(engine).is_dir():
                llm = LLM.from_artifacts(engine)
                source = "artifacts"
            else:
                llm = LLM(engine)  # HF 모델 id
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
                "devices": cfg.devices,
                "sampling_defaults": sampling_defaults,
                "extra": dict(cfg.extra or {}),
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
