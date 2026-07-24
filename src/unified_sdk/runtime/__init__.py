"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This QB-only worktree exposes only the Mobilint ARISE (QB) adapter.
"""
from .api import (  # re-export
    create_runtime,
    create_runtime_LLM,
    create_runtime_llm,
    destroy_runtime,
    destroy_runtime_LLM,
    destroy_runtime_llm,
    infer,
    infer_LLM,
    infer_llm,
)

# Adapter auto-registration
from . import qb_runtime as _qb  # noqa: F401
