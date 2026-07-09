"""
Unified SDK (QB-only worktree)
==============================

A unified SDK for compiling and running AI models on Mobilint ARISE (QB) NPUs.

Structure:
 - build:     Model compilation modules (qubee -> .mxq)
 - runtime:   Runtime creation and inference modules (qbruntime / QB-RUNTIME)
 - backends:  Backend adapters (Mobilint ARISE / QB only in this worktree)
 - frontends: Model import and conversion (ONNX, PyTorch, etc.)
"""

__version__ = "0.1.0"
