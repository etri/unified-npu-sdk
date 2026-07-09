"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model compilation interfaces.
Each backend registers its build adapter in the registry at import time.

This QB-only worktree exposes only the Mobilint ARISE (QB) adapter.
"""

from .api import build_unified  # Re-export high-level API

# Internal adapters (auto-registration)
from . import qb_build as _qb  # noqa: F401
