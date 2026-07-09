from __future__ import annotations
from typing import Any

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import RuntimeConfig, RuntimeHandle

# Adapter auto-registration
from . import rngd_runtime as _rngd  # noqa: F401


def create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)


def infer(rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    """RNGD: LLM text generation. `prompt` is a str or list[str]; returns
    generated text (str) or list[str]. Sampling params may be overridden per call."""
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, prompt, **overrides)


def generate(rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    """Readability alias of `infer` for the LLM backend."""
    return infer(rh, prompt, **overrides)


def destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)
