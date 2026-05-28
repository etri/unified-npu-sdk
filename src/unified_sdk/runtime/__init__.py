"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This RBLN-only worktree exposes only the RBLN adapter.
"""
from .api import create_runtime, infer, destroy_runtime  # re-export

# Adapter auto-registration
from . import rbln_runtime as _rbln  # noqa: F401
