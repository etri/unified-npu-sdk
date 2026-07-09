from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

BuildBackendName = Literal["tensorrt"]
RuntimeBackendName = Literal["tensorrt"]

# int8 은 calibrator 가 필요하다 (extra["int8_calibrator"]). 없으면 build 어댑터가 명시적으로 거부한다.
Precision = Literal["fp32", "fp16", "int8"]

@dataclass
class BuildConfig:
    backend: BuildBackendName
    model_or_path: str | Path                # ONNX 경로
    out_dir: str | Path = "build_output"
    model_name: str = "model"
    precision: Precision = "fp16"
    # TensorRT dynamic shape optimization profile
    input_name: str = "input"
    min_input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    opt_input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    max_input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    extra: Optional[Dict[str, Any]] = None   # workspace_mib, int8_calibrator, strict_types 등

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
    use_execute_v3: bool = True              # TRT 8.5+/10 의 권장 실행 경로
    extra: Optional[Dict[str, Any]] = None   # device, allow_dynamic_shape 등

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    # 백엔드 전용 객체/버퍼 캐리어 (engine, context, stream, device buffers ...)
    ctx: Dict[str, Any] = field(default_factory=dict)
