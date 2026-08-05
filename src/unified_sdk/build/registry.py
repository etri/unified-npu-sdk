from __future__ import annotations
from typing import Dict, Protocol

from unified_sdk.types import BuildConfig, BuildResult, LLMBuildConfig

class BuildAdapter(Protocol):
    name: str
    def build(self, cfg: BuildConfig) -> BuildResult: ...


class LLMBuildAdapter(Protocol):
    name: str
    def build(self, cfg: LLMBuildConfig) -> BuildResult: ...

_REGISTRY: Dict[str, BuildAdapter] = {}
_LLM_REGISTRY: Dict[str, LLMBuildAdapter] = {}

def register(adapter: BuildAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def register_llm(adapter: LLMBuildAdapter) -> None:
    _LLM_REGISTRY[adapter.name] = adapter

def get_builder(name: str) -> BuildAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Build backend '{name}' not registered. Available: {list(_REGISTRY)}")


def get_llm_builder(name: str) -> LLMBuildAdapter:
    try:
        return _LLM_REGISTRY[name]
    except KeyError:
        raise ValueError(f"LLM build backend '{name}' not registered. Available: {list(_LLM_REGISTRY)}")
