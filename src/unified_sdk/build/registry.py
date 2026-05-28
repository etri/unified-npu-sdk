from __future__ import annotations
from typing import Dict, Protocol
from unified_sdk.types import BuildConfig, BuildResult

class BuildAdapter(Protocol):
    name: str
    def build(self, cfg: BuildConfig) -> BuildResult: ...

_REGISTRY: Dict[str, BuildAdapter] = {}

def register(adapter: BuildAdapter) -> None:
    _REGISTRY[adapter.name] = adapter

def get_builder(name: str) -> BuildAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Build backend '{name}' not registered. Available: {list(_REGISTRY)}")

