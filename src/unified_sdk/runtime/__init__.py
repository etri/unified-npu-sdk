"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with compiled model.
"""
from .api import create_runtime, infer, destroy_runtime  # re-export

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
