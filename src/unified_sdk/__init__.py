"""
Unified SDK (TensorRT-only worktree)
====================================

A unified SDK for compiling and running AI models on NVIDIA GPUs via TensorRT / TensorRT-LLM.

Structure:
 - build:     Model compilation modules (ONNX -> TensorRT .engine, model -> TensorRT-LLM artifact dir)
 - runtime:   Runtime creation and inference modules (TensorRT + PyCUDA, TensorRT-LLM generate)
 - backends:  Backend adapters (TensorRT only in this worktree)
 - frontends: Model import and conversion (PyTorch, ONNX, etc.)

벤더 SDK(`tensorrt`, `pycuda`)는 어댑터 메서드 내부에서 lazy import 하므로,
GPU/TensorRT 가 없는 환경에서도 `import unified_sdk` 자체는 성공한다.
"""

__version__ = "0.1.0"
