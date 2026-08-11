from __future__ import annotations

from pathlib import Path
from typing import Any

from unified_sdk.frontends.types import (
    PreparedWarboyBuildInput,
    PreparedWarboyCompileSource,
    ProvidedWarboyArtifact,
)


def prepare_warboy_build_input(model_or_path: Any, enf_path: str | Path) -> PreparedWarboyBuildInput:
    output_path = Path(enf_path).expanduser().resolve()
    source_path = Path(model_or_path).expanduser().resolve() if isinstance(model_or_path, (str, Path)) else None

    if source_path is not None and source_path.suffix == ".enf":
        return PreparedWarboyBuildInput(
            kind="provided_artifact",
            provided_artifact=ProvidedWarboyArtifact(
                source_path=source_path,
                destination_path=output_path,
            ),
        )

    return PreparedWarboyBuildInput(
        kind="compile_source",
        compile_source=PreparedWarboyCompileSource(
            source=model_or_path,
            source_label="quantized_onnx",
        ),
    )
