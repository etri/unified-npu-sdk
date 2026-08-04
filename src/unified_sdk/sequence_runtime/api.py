from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from unified_sdk.sequence_runtime.registry import get_runtime
from unified_sdk.types import SequenceBatchParam, SequenceRuntimeConfig, SequenceRuntimeHandle

# Adapter auto-registration
from . import qb_sequence_runtime as _qb  # noqa: F401


def create_sequence_runtime(cfg: SequenceRuntimeConfig) -> SequenceRuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)


def infer_sequence(
    rh: SequenceRuntimeHandle,
    input_array: Any,
    *,
    cache_size: int = 0,
    batch_params: Optional[Sequence[SequenceBatchParam]] = None,
) -> Any:
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, input_array, cache_size=cache_size, batch_params=batch_params)


def destroy_sequence_runtime(rh: SequenceRuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)


def describe_sequence_runtime_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK sequence-runtime API mapping for this backend."""
    return _qb.describe_api_mapping()
