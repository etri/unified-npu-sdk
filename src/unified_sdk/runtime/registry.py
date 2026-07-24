from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, Sequence
from unified_sdk.types import BatchParam, RuntimeConfig, RuntimeHandle

class RuntimeAdapter(Protocol):
    name: str
    def create(self, cfg: RuntimeConfig) -> RuntimeHandle: ...
    def infer(
        self,
        rh: RuntimeHandle,
        input_array,
        *,
        cache_size: int = 0,
        batch_params: Optional[Sequence[BatchParam]] = None,
    ) -> Any: ...
    def destroy(self, rh: RuntimeHandle) -> None: ...

_REGISTRY: Dict[str, RuntimeAdapter] = {}

def register(adapter: RuntimeAdapter) -> None:
    _REGISTRY[adapter.name] = adapter

def get_runtime(name: str) -> RuntimeAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Runtime backend '{name}' not registered. Available: {list(_REGISTRY)}")
