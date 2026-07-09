from __future__ import annotations
from typing import Any, Dict

import numpy as np

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import RuntimeConfig, RuntimeHandle

# Adapter auto-registration
from . import tensorrt_runtime as _tensorrt  # noqa: F401


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
    return _tensorrt.describe_api_mapping()
