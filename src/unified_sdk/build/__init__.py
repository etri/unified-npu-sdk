"""
LLM build entrypoints for the RNGD-only worktree.

Preferred public surface:
 - `build_unified_LLM(cfg)` for RNGD LLM fetch / FXB build workflows

Compatibility surface:
 - `build_unified(cfg)` remains available for callers that still use the
   generic name, but this branch should be read as an LLM capability branch.
"""

from .api import build_unified, build_unified_LLM  # Re-export public API

# Internal adapters (auto-registration)
from . import rngd_build as _rngd  # noqa: F401
