from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PreparedRBLNSourceKind = Literal["provided_artifact", "compile_source"]
ResolvedRBLNVisionBuildKind = Literal[
    "provided_artifact",
    "compiled_dir",
    "optimum_source_model",
    "torch_model",
    "onnx_restore",
]


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
