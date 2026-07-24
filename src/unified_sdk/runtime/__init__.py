"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This TensorRT-only worktree exposes TensorRT vision and TensorRT-LLM adapters.
"""
from .api import (
    create_runtime,
    create_runtime_LLM,
    destroy_runtime,
    destroy_runtime_LLM,
    generate_LLM,
    infer,
    infer_LLM,
)  # re-export

# Adapter auto-registration
from . import tensorrt_runtime as _tensorrt  # noqa: F401
from . import tensorrt_llm_runtime as _tensorrt_llm  # noqa: F401
