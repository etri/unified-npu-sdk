from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ResolvedRNGDBuildKind = Literal["model_id", "local_model_path", "prebuilt_artifact_dir", "fxb_build_source"]


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
