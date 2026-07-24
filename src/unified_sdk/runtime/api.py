from __future__ import annotations
from typing import Any, Dict

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import RuntimeConfig, RuntimeHandle

# Adapter auto-registration
from . import rngd_runtime as _rngd  # noqa: F401


def _create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)


def create_runtime_LLM(cfg: RuntimeConfig) -> RuntimeHandle:
    """LLM-specific creation entrypoint for backend-specific text generation runtimes."""
    return _create_runtime(cfg)


def _infer(rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    """RNGD: LLM text generation. `prompt` is a str or list[str]; returns
    generated text (str) or list[str]. Sampling params may be overridden per call."""
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, prompt, **overrides)


def infer_LLM(rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    """LLM-specific inference/generation entrypoint."""
    return _infer(rh, prompt, **overrides)


def generate_LLM(rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    """Readability alias of `infer_LLM` for the LLM backend."""
    return infer_LLM(rh, prompt, **overrides)


def _destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)


def destroy_runtime_LLM(rh: RuntimeHandle) -> None:
    """LLM-specific destroy entrypoint."""
    return _destroy_runtime(rh)


create_runtime_llm = create_runtime_LLM
infer_llm = infer_LLM
generate_llm = generate_LLM
destroy_runtime_llm = destroy_runtime_LLM


def describe_runtime_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK runtime API mapping for this backend."""
    return _rngd.describe_api_mapping()
