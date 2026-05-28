"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model compilation interfaces.
Each backend registers its build adapter in the registry at import time.

This RBLN-only worktree exposes only the RBLN adapter.
"""

from .api import build_unified  # Re-export high-level API

# Internal adapters (auto-registration)
from . import rbln_build as _rbln  # noqa: F401
