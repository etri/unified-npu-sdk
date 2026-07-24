from __future__ import annotations
from typing import Any, Dict

import numpy as np

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import LLMRuntimeConfig, LLMRuntimeHandle, RuntimeConfig, RuntimeHandle

# Adapter auto-registration
from . import rbln_runtime as _rbln  # noqa: F401
from . import rbln_llm_runtime as _rbln_llm  # noqa: F401


def create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)


def infer(rh: RuntimeHandle, input_array: "np.ndarray") -> "np.ndarray":
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, input_array)


def destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)


def describe_runtime_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK runtime API mapping for this backend."""
    return _rbln.describe_api_mapping()


def create_runtime_LLM(cfg: LLMRuntimeConfig) -> LLMRuntimeHandle:
    return _rbln_llm.create_llm(cfg)


def generate_LLM(rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    return _rbln_llm.generate_llm(rh, prompt, **overrides)


def infer_LLM(rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any:
    return generate_LLM(rh, prompt, **overrides)


def destroy_runtime_LLM(rh: LLMRuntimeHandle) -> None:
    return _rbln_llm.destroy_llm(rh)


create_runtime_llm = create_runtime_LLM
generate_llm = generate_LLM
infer_llm = infer_LLM
destroy_runtime_llm = destroy_runtime_LLM


def describe_runtime_api_mapping_LLM() -> Dict[str, Any]:
    return _rbln_llm.describe_api_mapping()
