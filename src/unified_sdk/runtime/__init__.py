"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with compiled model.
"""
from .api import create_runtime, infer, destroy_runtime  # re-export
# 내부 어댑터 자동 등록
from . import tensorrt_runtime as _tensorrt  # noqa: F401
