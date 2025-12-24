from __future__ import annotations
from typing import Any
import numpy as np

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import RuntimeConfig, RuntimeHandle, RuntimeBackendName


# 내부 어댑터 자동 등록
# from . import tensorrt_runtime as _tensorrt  # noqa: F401
# from . import rbln_runtime as _rbln          # noqa: F401



import warnings

try:
    from . import tensorrt_runtime as _tensorrt  # noqa: F401
except Exception as e:
    warnings.warn(f"TensorRT backend disabled: {e!r}")
    _tensorrt = None

try:
    from . import rbln_runtime as _rbln  # noqa: F401
except Exception as e:
    warnings.warn(f"RBLN backend disabled: {e!r}")
    _rbln = None
    


def create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)

def infer(rh: RuntimeHandle, input_array: "np.ndarray") -> "np.ndarray":
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, input_array)

def destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)

