from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from unified_sdk.options import RNGDBuildOptions, RNGDRuntimeOptions

BuildBackendName = Literal["rngd"]
RuntimeBackendName = Literal["rngd"]

# RNGD 는 LLM 스택이라 vision 백엔드(rbln/qb/warboy)의 input_shape/precision 대신
# LLM 준비/서빙에 맞는 필드를 노출한다.

@dataclass
class BuildConfig:
    backend: BuildBackendName
    # HuggingFace 모델 id (예: 'furiosa-ai/Qwen2.5-0.5B-Instruct') 또는 로컬 모델 경로
    model_or_path: Any
    out_dir: str | Path = "artifacts"
    model_name: str = "model"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    backend_options: RNGDBuildOptions | Dict[str, Any] | None = None
    extra: Optional[Dict[str, Any]] = None  # legacy fallback: build_mode(fetch|fxb_build), dry_run, optim_level 등

@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str            # HF 모델 id 또는 FXB 파일 경로
    meta_data: Dict[str, Any]


@dataclass
class LLMBuildConfig(BuildConfig):
    """Explicit LLM capability config for the RNGD-only worktree."""

@dataclass
class RuntimeConfig:
    backend: RuntimeBackendName
    engine_path: str | Path             # HF 모델 id 또는 로컬 모델 경로
    # 기본 SamplingParams (호출별로 override 가능)
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    min_tokens: int = 0
    fxb_path: Optional[str | Path] = None     # legacy compatibility: prefer backend_options.fxb_path
    devices: Optional[str] = None             # legacy compatibility: prefer backend_options.devices
    backend_options: RNGDRuntimeOptions | Dict[str, Any] | None = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class LLMRuntimeConfig(RuntimeConfig):
    """Explicit LLM runtime config for the RNGD-only worktree."""

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)
