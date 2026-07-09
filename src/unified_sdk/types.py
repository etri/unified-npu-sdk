from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

BuildBackendName = Literal["warboy"]
RuntimeBackendName = Literal["warboy"]

# Warboy 실행 모델(.enf)은 quantized ONNX 를 furiosa-compiler 가 int8 로 컴파일한 결과다.
Precision = Literal["int8"]

@dataclass
class BuildConfig:
    backend: BuildBackendName
    model_or_path: Any                      # quantized ONNX 경로 / 기존 .enf 경로
    out_dir: str | Path = "builds"
    model_name: str = "model"
    precision: Precision = "int8"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    # (선택) f32 ONNX 를 직접 넘길 때 furiosa.quantizer 로 양자화하기 위한 calibration 메타.
    # quantized ONNX 를 이미 넘기면 사용하지 않는다.
    calib_data_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None  # target_npu, target_ir, compiler_config 등

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
    extra: Optional[Dict[str, Any]] = None  # device, target_npu, allow_dynamic_shape 등

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)
