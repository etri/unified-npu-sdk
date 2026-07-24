"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a prepared model.

This RNGD-only worktree exposes only the FuriosaAI RNGD (furiosa-llm) adapter.
For RNGD, inference is LLM text generation (prompt -> text). This worktree
exports only the explicit LLM API set:
`create_runtime_LLM`, `infer_LLM`, `generate_LLM`, `destroy_runtime_LLM`.
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
