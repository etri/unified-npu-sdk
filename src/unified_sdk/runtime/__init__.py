"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This TensorRT-only worktree exposes only the TensorRT adapter.
"""
from .api import create_runtime, infer, destroy_runtime  # re-export

# Adapter auto-registration
from . import tensorrt_runtime as _tensorrt  # noqa: F401
