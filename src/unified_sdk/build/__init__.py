"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model compilation interfaces.
Each backend registers its build adapter in the registry at import time.

This Warboy-only worktree exposes only the FuriosaAI Warboy adapter.
"""

from .api import build_unified  # Re-export high-level API

# Internal adapters (auto-registration)
from . import warboy_build as _warboy  # noqa: F401
