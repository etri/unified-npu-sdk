"""
Unified SDK (QB-only worktree)
==============================

A unified SDK for compiling and running AI models on Mobilint ARISE (QB) NPUs.

Structure:
 - build:     Model compilation modules (qubee -> .mxq)
 - runtime:   Vision runtime modules (qbruntime / QB-RUNTIME)
 - sequence_runtime: Low-level sequence runtime modules (cache-aware qbruntime path)
 - backends:  Backend adapters (Mobilint ARISE / QB only in this worktree)
 - frontends: Model import and conversion (ONNX, PyTorch, etc.)

Preferred public surface:
 - `unified_sdk.options`: typed QB backend options
 - `unified_sdk.runtime`: vision runtime capability
 - `unified_sdk.sequence_runtime`: low-level sequence runtime capability
"""

__version__ = "0.1.0"
