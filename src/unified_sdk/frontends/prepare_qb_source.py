from __future__ import annotations

from pathlib import Path
from typing import Any

from unified_sdk.frontends.types import (
    PreparedQBBuildInput,
    PreparedQBCompileSource,
    ProvidedQBArtifact,
)


def _looks_like_mxq(model_or_path: Any) -> bool:
    return isinstance(model_or_path, (str, Path)) and str(model_or_path).endswith(".mxq")


def prepare_qb_build_input(model_or_path: Any, destination_path: str | Path) -> PreparedQBBuildInput:
    destination = Path(destination_path)
    if _looks_like_mxq(model_or_path):
        source_path = Path(model_or_path).expanduser().resolve()
        return PreparedQBBuildInput(
            kind="provided_artifact",
            provided_artifact=ProvidedQBArtifact(
                source_path=source_path,
                destination_path=destination,
            ),
        )

    source_label = type(model_or_path).__name__
    if isinstance(model_or_path, (str, Path)):
        source_label = str(Path(model_or_path).expanduser())
    return PreparedQBBuildInput(
        kind="compile_source",
        compile_source=PreparedQBCompileSource(
            source=model_or_path,
            source_label=source_label,
        ),
    )
