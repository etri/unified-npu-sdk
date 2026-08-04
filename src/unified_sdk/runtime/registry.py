from __future__ import annotations
from typing import Any, Dict, Protocol

from unified_sdk.types import RuntimeConfig, RuntimeHandle

class RuntimeAdapter(Protocol):
    name: str
    def create(self, cfg: RuntimeConfig) -> RuntimeHandle: ...
    def infer(self, rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any: ...
    def generate(self, rh: RuntimeHandle, prompt: Any, **overrides: Any) -> Any: ...
    def destroy(self, rh: RuntimeHandle) -> None: ...

_REGISTRY: Dict[str, RuntimeAdapter] = {}

def register(adapter: RuntimeAdapter) -> None:
    _REGISTRY[adapter.name] = adapter

def get_runtime(name: str) -> RuntimeAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Runtime backend '{name}' not registered. Available: {list(_REGISTRY)}")
