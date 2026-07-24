"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model compilation interfaces.
Each backend registers its build adapter in the registry at import time.

This RBLN-only worktree exposes:
 - vision build: build_unified(cfg)
 - LLM build:    build_unified_LLM(cfg)
"""

from .api import build_unified, build_unified_LLM  # Re-export high-level API

# Internal adapters (auto-registration)
from . import rbln_build as _rbln  # noqa: F401
from . import rbln_llm_build as _rbln_llm  # noqa: F401
