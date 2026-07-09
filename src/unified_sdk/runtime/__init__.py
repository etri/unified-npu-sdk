"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This QB-only worktree exposes only the Mobilint ARISE (QB) adapter.
"""
from .api import create_runtime, infer, destroy_runtime  # re-export

# Adapter auto-registration
from . import qb_runtime as _qb  # noqa: F401
