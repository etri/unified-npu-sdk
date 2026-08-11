from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PreparedQBSourceKind = Literal["provided_artifact", "compile_source"]
ResolvedQBBuildKind = Literal["model_zoo_fetch", "provided_artifact", "local_onnx", "weights_export"]
PreparedWarboySourceKind = Literal["provided_artifact", "compile_source"]
ResolvedWarboyBuildKind = Literal["provided_artifact", "local_enf", "model_zoo_enf", "quantized_onnx"]
ResolvedRNGDBuildKind = Literal["model_id", "local_model_path", "prebuilt_artifact_dir", "fxb_build_source"]
PreparedRBLNSourceKind = Literal["provided_artifact", "compile_source"]
PreparedRBLNLLMBuildKind = Literal["runtime_model_ref", "artifact_build"]
ResolvedRBLNVisionBuildKind = Literal[
    "provided_artifact",
    "compiled_dir",
    "optimum_source_model",
    "torch_model",
    "pth_restore",
    "onnx_restore",
]
ResolvedRBLNLLMBuildKind = Literal["runtime_model_ref", "artifact_build"]
PreparedTensorRTVisionSourceKind = Literal["provided_artifact", "compile_source"]
ResolvedTensorRTVisionBuildKind = Literal["provided_artifact", "onnx_path", "torchvision_export", "pth_export"]
PreparedTensorRTLLMFetchKind = Literal["runtime_model_ref"]
ResolvedTensorRTLLMFetchKind = Literal["runtime_model_ref"]
PreparedTensorRTLLMBuildKind = Literal["artifact_build"]
ResolvedTensorRTLLMBuildKind = Literal["artifact_build"]
PreparedTensorRTLLMSourceKind = Literal["model_id", "local_model_path", "local_artifact_dir", "local_checkpoint_dir"]
PreparedTensorRTLLMCompileVariant = Literal["model_ref_api", "checkpoint_dir_cli"]


@dataclass(frozen=True)
class QBFrontendBuildRequest:
    model_name: str
    models_dir: Path
    product: str
    core_mode: str
    from_pth: Path | None = None
    from_onnx: Path | None = None
    provided_mxq: Path | None = None
    export_onnx_path: Path | None = None
    input_name: str = "input"
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    require_mxq: bool = False


@dataclass(frozen=True)
class ProvidedQBArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedQBCompileSource:
    source: Any
    source_label: str


@dataclass(frozen=True)
class PreparedQBBuildInput:
    kind: PreparedQBSourceKind
    provided_artifact: ProvidedQBArtifact | None = None
    compile_source: PreparedQBCompileSource | None = None


@dataclass(frozen=True)
class ResolvedQBBuildRequest:
    model_or_path: str
    source_description: str
    kind: ResolvedQBBuildKind


@dataclass(frozen=True)
class WarboyFrontendBuildRequest:
    model_name: str
    models_dir: Path
    target_npu: str
    provided_enf: Path | None = None
    from_onnx: Path | None = None
    require_enf: bool = False


@dataclass(frozen=True)
class ProvidedWarboyArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedWarboyCompileSource:
    source: Any
    source_label: str


@dataclass(frozen=True)
class PreparedWarboyBuildInput:
    kind: PreparedWarboySourceKind
    provided_artifact: ProvidedWarboyArtifact | None = None
    compile_source: PreparedWarboyCompileSource | None = None


@dataclass(frozen=True)
class PreparedWarboyRuntimeInput:
    batch: Any
    source_description: str
    expected_dtype: str | None
    actual_dtype: str | None
    contexts: Any = None
    model_helper: Any = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedWarboyBuildRequest:
    model_or_path: str
    source_description: str
    kind: ResolvedWarboyBuildKind
    prepared_input: PreparedWarboyBuildInput


@dataclass(frozen=True)
class RNGDFrontendBuildRequest:
    model_or_path: str | Path
    out_dir: Path
    model_name: str
    build_mode: Literal["fetch", "fxb_build"]


@dataclass(frozen=True)
class ResolvedRNGDBuildRequest:
    model_ref: str
    output_path: Path | None
    kind: ResolvedRNGDBuildKind
    source_description: str


@dataclass(frozen=True)
class RBLNVisionFrontendBuildRequest:
    model_name: str
    models_dir: Path
    compiled_model_ref: str | None = None
    provided_rbln: Path | None = None
    from_onnx: Path | None = None
    weights_path: Path | None = None
    model_zoo_model: str | None = None
    pretrained: bool = False
    require_weights: bool = False


@dataclass(frozen=True)
class ProvidedRBLNArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedRBLNCompileSource:
    source: Any
    source_label: str
    provenance_kind: ResolvedRBLNVisionBuildKind
    provenance_detail: str
    source_path: Path | None = None
    compile_frontend: Literal["rebel", "optimum_image_classification"] = "rebel"
    source_cache_dir: Path | None = None


@dataclass(frozen=True)
class PreparedRBLNVisionBuildInput:
    kind: PreparedRBLNSourceKind
    provided_artifact: ProvidedRBLNArtifact | None = None
    compile_source: PreparedRBLNCompileSource | None = None


@dataclass(frozen=True)
class ResolvedRBLNVisionBuildRequest:
    model_or_path: Any
    source_description: str
    kind: ResolvedRBLNVisionBuildKind
    prepared_input: PreparedRBLNVisionBuildInput


@dataclass(frozen=True)
class RBLNLLMFrontendBuildRequest:
    model_ref: str | Path
    out_dir: Path
    model_name: str
    build_mode: Literal["fetch", "optimum_compile"] = "fetch"


@dataclass(frozen=True)
class PreparedRBLNLLMBuildInput:
    kind: PreparedRBLNLLMBuildKind
    model_ref: str
    artifact_dir: Path | None = None


@dataclass(frozen=True)
class ResolvedRBLNLLMBuildRequest:
    source_description: str
    kind: ResolvedRBLNLLMBuildKind
    prepared_input: PreparedRBLNLLMBuildInput


@dataclass(frozen=True)
class TensorRTVisionFrontendBuildRequest:
    model_name: str
    models_dir: Path
    out_dir: Path
    precision: Literal["fp32", "fp16", "int8"] = "fp32"
    provided_engine: Path | None = None
    onnx_path: Path | None = None
    weights_path: Path | None = None
    export_onnx_path: Path | None = None
    model_zoo_model: str | None = None
    pretrained: bool = False
    require_onnx: bool = False
    input_name: str = "input"
    min_input_shape: tuple[int, ...] = (1, 3, 224, 224)
    opt_input_shape: tuple[int, ...] = (1, 3, 224, 224)
    max_input_shape: tuple[int, ...] = (1, 3, 224, 224)


@dataclass(frozen=True)
class ProvidedTensorRTArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedTensorRTCompileSource:
    source_path: Path
    source_label: str
    provenance_kind: ResolvedTensorRTVisionBuildKind
    provenance_detail: str
    input_name: str
    min_input_shape: tuple[int, ...]
    opt_input_shape: tuple[int, ...]
    max_input_shape: tuple[int, ...]


@dataclass(frozen=True)
class PreparedTensorRTVisionBuildInput:
    kind: PreparedTensorRTVisionSourceKind
    provided_artifact: ProvidedTensorRTArtifact | None = None
    compile_source: PreparedTensorRTCompileSource | None = None


@dataclass(frozen=True)
class ResolvedTensorRTVisionBuildRequest:
    model_or_path: str
    source_description: str
    kind: ResolvedTensorRTVisionBuildKind
    prepared_input: PreparedTensorRTVisionBuildInput


@dataclass(frozen=True)
class TensorRTLLMFrontendFetchRequest:
    model_ref: str | Path


@dataclass(frozen=True)
class PreparedTensorRTLLMFetchInput:
    kind: PreparedTensorRTLLMFetchKind
    model_ref: str
    source_kind: PreparedTensorRTLLMSourceKind = "model_id"
    source_path: Path | None = None


@dataclass(frozen=True)
class ResolvedTensorRTLLMFetchRequest:
    source_description: str
    kind: ResolvedTensorRTLLMFetchKind
    prepared_input: PreparedTensorRTLLMFetchInput


@dataclass(frozen=True)
class TensorRTLLMFrontendBuildRequest:
    model_ref: str | Path
    out_dir: Path
    model_name: str


@dataclass(frozen=True)
class PreparedTensorRTLLMBuildInput:
    kind: PreparedTensorRTLLMBuildKind
    model_ref: str
    source_kind: PreparedTensorRTLLMSourceKind = "model_id"
    source_path: Path | None = None
    artifact_dir: Path | None = None
    compile_variant: PreparedTensorRTLLMCompileVariant | None = None
    checkpoint_dir: Path | None = None


@dataclass(frozen=True)
class ResolvedTensorRTLLMBuildRequest:
    source_description: str
    kind: ResolvedTensorRTLLMBuildKind
    prepared_input: PreparedTensorRTLLMBuildInput
