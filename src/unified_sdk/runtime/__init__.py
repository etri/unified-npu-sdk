"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This Warboy-only worktree exposes only the FuriosaAI Warboy adapter.
"""
from .api import create_runtime, infer, destroy_runtime, describe_runtime_api_mapping  # re-export

# Adapter auto-registration
from . import warboy_runtime as _warboy  # noqa: F401
