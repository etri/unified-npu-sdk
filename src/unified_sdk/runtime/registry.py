from __future__ import annotations
from typing import Any, Dict, Protocol

from unified_sdk.types import LLMRuntimeHandle, RuntimeConfig

class RuntimeAdapter(Protocol):
    name: str
    def create(self, cfg: RuntimeConfig) -> LLMRuntimeHandle: ...
    def infer(self, rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any: ...
    def generate(self, rh: LLMRuntimeHandle, prompt: Any, **overrides: Any) -> Any: ...
    def destroy(self, rh: LLMRuntimeHandle) -> None: ...

_REGISTRY: Dict[str, RuntimeAdapter] = {}

def register(adapter: RuntimeAdapter) -> None:
    _REGISTRY[adapter.name] = adapter

def get_runtime(name: str) -> RuntimeAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Runtime backend '{name}' not registered. Available: {list(_REGISTRY)}")
