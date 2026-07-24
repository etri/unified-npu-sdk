"""
unified_sdk.runtime
-------------------
Module responsible for creating runtime instances and performing inference
with a compiled model.

This RBLN-only worktree exposes:
 - vision runtime: create_runtime / infer / destroy_runtime
 - LLM runtime:    create_runtime_LLM / generate_LLM / destroy_runtime_LLM
"""
from .api import (
    create_runtime,
    create_runtime_LLM,
    create_runtime_llm,
    destroy_runtime,
    destroy_runtime_LLM,
    destroy_runtime_llm,
    generate_LLM,
    generate_llm,
    infer,
    infer_LLM,
    infer_llm,
)  # re-export

# Adapter auto-registration
from . import rbln_runtime as _rbln  # noqa: F401
from . import rbln_llm_runtime as _rbln_llm  # noqa: F401
