from __future__ import annotations
from typing import Any, Dict, Optional, Sequence

import numpy as np

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import BatchParam, RuntimeConfig, RuntimeHandle

# Adapter auto-registration
from . import qb_runtime as _qb  # noqa: F401


def create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)


def infer(
    rh: RuntimeHandle,
    input_array: "np.ndarray",
    *,
    cache_size: int = 0,
    batch_params: Optional[Sequence[BatchParam]] = None,
) -> Any:
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, input_array, cache_size=cache_size, batch_params=batch_params)


def destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)


def describe_runtime_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK runtime API mapping for this backend."""
    return _qb.describe_api_mapping()
