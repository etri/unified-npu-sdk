"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model compilation interfaces.
Each backend registers its build adapter in the registry at import time.

This TensorRT-only worktree exposes TensorRT vision and TensorRT-LLM adapters.
"""

from .api import build_unified, build_unified_LLM, fetch_unified_LLM  # Re-export high-level APIs

# Internal adapters (auto-registration)
from . import tensorrt_build as _tensorrt  # noqa: F401
from . import tensorrt_llm_build as _tensorrt_llm  # noqa: F401
