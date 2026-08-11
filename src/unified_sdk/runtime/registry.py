from __future__ import annotations

from typing import Dict, Protocol

from unified_sdk.types import LLMRuntimeConfig, LLMRuntimeHandle, RuntimeConfig, RuntimeHandle


class RuntimeAdapter(Protocol):
    name: str

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle: ...
    def infer(self, rh: RuntimeHandle, input_array) -> "np.ndarray": ...
    def destroy(self, rh: RuntimeHandle) -> None: ...


class LLMRuntimeAdapter(Protocol):
    name: str

    def create(self, cfg: LLMRuntimeConfig) -> LLMRuntimeHandle: ...
    def infer(self, rh: LLMRuntimeHandle, prompt, **overrides): ...
    def destroy(self, rh: LLMRuntimeHandle) -> None: ...


_REGISTRY: Dict[str, RuntimeAdapter] = {}
_LLM_REGISTRY: Dict[str, LLMRuntimeAdapter] = {}


def register(adapter: RuntimeAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def register_llm(adapter: LLMRuntimeAdapter) -> None:
    _LLM_REGISTRY[adapter.name] = adapter


def get_runtime(name: str) -> RuntimeAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Runtime backend '{name}' not registered. Available: {list(_REGISTRY)}")


def get_llm_runtime(name: str) -> LLMRuntimeAdapter:
    try:
        return _LLM_REGISTRY[name]
    except KeyError:
        raise ValueError(f"LLM runtime backend '{name}' not registered. Available: {list(_LLM_REGISTRY)}")
