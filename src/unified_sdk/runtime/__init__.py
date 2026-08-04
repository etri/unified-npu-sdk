"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This QB-only worktree exposes only the Mobilint ARISE (QB) adapter.
"""
from .api import (  # re-export
    create_runtime,
    destroy_runtime,
    infer,
)

# Adapter auto-registration
from . import qb_runtime as _qb  # noqa: F401
