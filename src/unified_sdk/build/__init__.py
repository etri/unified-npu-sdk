"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model preparation interfaces.
Each backend registers its build adapter in the registry at import time.

This RNGD-only worktree exposes only the FuriosaAI RNGD (furiosa-llm) adapter.
"""

from .api import build_unified, build_unified_LLM  # Re-export high-level API

# Internal adapters (auto-registration)
from . import rngd_build as _rngd  # noqa: F401
