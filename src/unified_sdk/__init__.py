"""
Unified SDK (Warboy-only worktree)
==================================

A unified SDK for compiling and running AI models on FuriosaAI Warboy NPUs.

Structure:
 - build:     Model compilation modules (furiosa-compiler -> .enf)
 - runtime:   Runtime creation and inference modules (furiosa.runtime)
 - backends:  Backend adapters (FuriosaAI Warboy only in this worktree)
 - frontends: Model import and conversion (ONNX, PyTorch, etc.)
"""

__version__ = "0.1.0"
