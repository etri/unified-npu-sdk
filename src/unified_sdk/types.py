from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

from unified_sdk.options import QBBuildOptions, QBVisionRuntimeOptions

BuildBackendName = Literal["qb"]
RuntimeBackendName = Literal["qb"]

@dataclass
class BuildConfig:
    backend: BuildBackendName
    model_or_path: Any                      # ONNX 경로 / torch.nn.Module / 기존 .mxq 경로
    out_dir: str | Path = "builds"
    model_name: str = "model"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    backend_options: QBBuildOptions | None = None

@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]

@dataclass
class RuntimeConfig:
    backend: RuntimeBackendName
    engine_path: str | Path
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    backend_options: QBVisionRuntimeOptions | None = None

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)
