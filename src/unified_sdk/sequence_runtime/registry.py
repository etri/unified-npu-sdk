from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence

from unified_sdk.sequence_runtime.types import (
    SequenceBatchParam,
    SequenceRuntimeConfig,
    SequenceRuntimeHandle,
)


class SequenceRuntimeAdapter(Protocol):
    name: str

    def create(self, cfg: SequenceRuntimeConfig) -> SequenceRuntimeHandle: ...

    def infer(
        self,
        rh: SequenceRuntimeHandle,
        input_array: Any,
        *,
        cache_size: int = 0,
        batch_params: Optional[Sequence[SequenceBatchParam]] = None,
    ) -> Any: ...

    def destroy(self, rh: SequenceRuntimeHandle) -> None: ...


_REGISTRY: Dict[str, SequenceRuntimeAdapter] = {}


def register(adapter: SequenceRuntimeAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_runtime(name: str) -> SequenceRuntimeAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Sequence low-level runtime backend '{name}' not registered. Available: {list(_REGISTRY)}"
        )
