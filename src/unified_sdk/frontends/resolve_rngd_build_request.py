from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.frontends.types import RNGDFrontendBuildRequest, ResolvedRNGDBuildRequest


_FRONTEND_PIPELINE = (
    "normalize_request",
    "classify_model_reference",
    "validate_build_mode",
    "resolve_output_path",
    "emit_resolved_build_request",
)
_FRONTEND_API_MAP = {
    "fetch": "resolve_rngd_build_request(request=RNGDFrontendBuildRequest(build_mode='fetch', ...))",
    "fxb_build": "resolve_rngd_build_request(request=RNGDFrontendBuildRequest(build_mode='fxb_build', ...))",
    "artifact_detection": "frontends.resolve_rngd_build_request(...) for local artifact/model directory classification",
}
_ARTIFACT_MARKERS = ("artifact.json", "binary_bundle.zip", "model_metadata.json")


def describe_frontend_api_mapping() -> Dict[str, Any]:
    return {
        "unified_frontend_api": "resolve_rngd_build_request(request=RNGDFrontendBuildRequest(...))",
        "capability_family": "llm.frontend-prepare-fetch",
        "pipeline": _FRONTEND_PIPELINE,
        "vendor_api_map": _FRONTEND_API_MAP,
    }


def detect_prebuilt_artifact_dir(model_ref: str) -> str | None:
    path = Path(model_ref)
    if not path.is_dir():
        return None

    markers = [name for name in _ARTIFACT_MARKERS if (path / name).exists()]
    if not markers:
        return None
    return ", ".join(markers)


def resolve_rngd_build_request(*, request: RNGDFrontendBuildRequest) -> ResolvedRNGDBuildRequest:
    model_ref = str(request.model_or_path).strip()
    if not model_ref:
        raise ValueError("RNGDFrontendBuildRequest.model_or_path must not be empty")

    out_dir = request.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(request.model_name).strip()
    if not model_name:
        raise ValueError("RNGDFrontendBuildRequest.model_name must not be empty")

    artifact_markers = detect_prebuilt_artifact_dir(model_ref)
    model_path = Path(model_ref)
    if artifact_markers:
        kind = "prebuilt_artifact_dir"
        source_description = f"local prebuilt artifact directory: {model_path.resolve()}"
    elif model_path.is_dir():
        kind = "local_model_path"
        source_description = f"local upstream/raw model directory: {model_path.resolve()}"
    else:
        kind = "model_id"
        source_description = f"Hugging Face model id or unresolved local reference: {model_ref}"

    if request.build_mode == "fetch":
        return ResolvedRNGDBuildRequest(
            model_ref=model_ref,
            output_path=None,
            kind=kind,
            source_description=source_description,
        )

    output_path = out_dir / model_name
    if output_path.suffix != ".fxb":
        output_path = output_path.with_suffix(".fxb")

    if artifact_markers:
        raise RuntimeError(
            "FXB build expects an upstream/raw Hugging Face model snapshot or a local model directory, "
            f"but {model_ref!r} looks like a prebuilt Furiosa artifact repo ({artifact_markers}). "
            "Use this path with the standard smoke (`model id/local artifact -> generate`) instead, "
            "or prepare an upstream model snapshot for custom FXB smoke."
        )

    return ResolvedRNGDBuildRequest(
        model_ref=model_ref,
        output_path=output_path,
        kind="fxb_build_source",
        source_description=f"custom FXB build source: {source_description}",
    )
