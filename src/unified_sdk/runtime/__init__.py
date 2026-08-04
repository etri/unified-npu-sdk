"""
unified_sdk.runtime
-------------------
Runtime entrypoints for the RNGD-only LLM capability.

This RNGD-only worktree exposes only the FuriosaAI RNGD (furiosa-llm) adapter.
Preferred public surface:
 - `create_runtime_LLM`
 - `generate_LLM`
 - `destroy_runtime_LLM`

Compatibility aliases:
 - `infer_LLM` remains available, but in this worktree it is an alias for
   text generation rather than a separate numpy-style inference contract.
"""
from .api import (
    create_runtime_LLM,
    create_runtime_llm,
    destroy_runtime_LLM,
    destroy_runtime_llm,
    generate_LLM,
    generate_llm,
    infer_LLM,
    infer_llm,
)  # re-export

# Adapter auto-registration
from . import rngd_runtime as _rngd  # noqa: F401
