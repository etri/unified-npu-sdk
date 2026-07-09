"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a prepared model.

This RNGD-only worktree exposes only the FuriosaAI RNGD (furiosa-llm) adapter.
For RNGD, inference is LLM text generation (prompt -> text); `generate` is a
readability alias of `infer`.
"""
from .api import create_runtime, infer, generate, destroy_runtime  # re-export

# Adapter auto-registration
from . import rngd_runtime as _rngd  # noqa: F401
