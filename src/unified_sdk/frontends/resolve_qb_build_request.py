from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.frontends.export_qb_onnx import export_supported_onnx_from_pth
from unified_sdk.frontends.qb_model_zoo import (
    find_local_mxq,
    find_model_zoo_mxq,
    normalize_mxq_into_models,
    trigger_model_zoo_fetch,
)
from unified_sdk.frontends.types import QBFrontendBuildRequest, ResolvedQBBuildRequest


_FRONTEND_PIPELINE = (
    "normalize_request",
    "resolve_weights_export_or_onnx_source",
    "resolve_local_or_model_zoo_artifact",
    "materialize_local_artifact",
    "emit_resolved_build_request",
)
_FRONTEND_API_MAP = {
    "weights_export": "frontends.export_supported_onnx_from_pth(weights_path, export_onnx_path, model_name, input_name, input_shape)",
    "local_onnx": "frontends.resolve_qb_build_request(request=QBFrontendBuildRequest(from_onnx=...))",
    "local_or_provided_mxq": "frontends.resolve_qb_build_request(request=QBFrontendBuildRequest(provided_mxq=...))",
    "model_zoo_fetch": "frontends.trigger_model_zoo_fetch(model_name, product, core_mode, models_dir)",
    "artifact_normalize": "frontends.normalize_mxq_into_models(mxq_path, models_dir, model_name)",
}


def describe_frontend_api_mapping() -> Dict[str, Any]:
    return {
        "unified_frontend_api": "resolve_qb_build_request(request=QBFrontendBuildRequest(...))",
        "capability_family": "vision.frontend-prepare-fetch",
        "pipeline": _FRONTEND_PIPELINE,
        "vendor_api_map": _FRONTEND_API_MAP,
    }


def _normalize_request(request: QBFrontendBuildRequest) -> QBFrontendBuildRequest:
    return QBFrontendBuildRequest(
        model_name=request.model_name,
        models_dir=request.models_dir.expanduser().resolve(),
        product=request.product,
        core_mode=request.core_mode,
        from_pth=request.from_pth.expanduser().resolve() if request.from_pth is not None else None,
        from_onnx=request.from_onnx.expanduser().resolve() if request.from_onnx is not None else None,
        provided_mxq=request.provided_mxq.expanduser().resolve() if request.provided_mxq is not None else None,
        export_onnx_path=request.export_onnx_path.expanduser().resolve() if request.export_onnx_path is not None else None,
        input_name=request.input_name,
        input_shape=request.input_shape,
        require_mxq=request.require_mxq,
    )


def _resolve_weights_export_request(request: QBFrontendBuildRequest) -> ResolvedQBBuildRequest | None:
    if request.from_pth is None:
        return None
    weights_path = request.from_pth
    if not weights_path.is_file():
        raise FileNotFoundError(f"PTH/PT weights not found: {weights_path}")
    resolved_export_path = request.export_onnx_path or (request.models_dir / f"{request.model_name}.onnx").resolve()
    onnx_path = export_supported_onnx_from_pth(
        weights_path=weights_path,
        export_onnx_path=resolved_export_path,
        model_name=request.model_name,
        input_name=request.input_name,
        input_shape=request.input_shape,
    )
    return ResolvedQBBuildRequest(
        model_or_path=str(onnx_path),
        source_description=f"local weights -> ONNX export -> compiler Python API compile: {weights_path} -> {onnx_path}",
        kind="weights_export",
    )


def _resolve_local_onnx_request(request: QBFrontendBuildRequest) -> ResolvedQBBuildRequest | None:
    if request.from_onnx is None:
        return None
    onnx_path = request.from_onnx
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    return ResolvedQBBuildRequest(
        model_or_path=str(onnx_path),
        source_description=f"local/custom ONNX -> compiler Python API compile: {onnx_path}",
        kind="local_onnx",
    )


def _resolve_fetch_request(request: QBFrontendBuildRequest) -> ResolvedQBBuildRequest:
    mxq = request.provided_mxq or find_local_mxq(request.models_dir, request.model_name)
    source_description = ""
    kind = "provided_artifact"
    if mxq is None:
        mxq = find_model_zoo_mxq(request.model_name, request.product, request.core_mode)
        if mxq is None:
            mxq = trigger_model_zoo_fetch(request.model_name, request.product, request.core_mode, request.models_dir)
        if mxq is not None:
            normalized_mxq = normalize_mxq_into_models(mxq, request.models_dir, request.model_name)
            source_description = f"standard fetch from official model zoo: {mxq} -> {normalized_mxq}"
            mxq = normalized_mxq
            kind = "model_zoo_fetch"

    if mxq is None:
        msg = (
            f"{request.models_dir} 또는 ~/.mblt_model_zoo/vision/{request.product}/{request.core_mode} 에서 "
            f"{request.model_name}*.mxq 를 찾지 못했습니다.\n"
            "표준 fetch는 ~/.mblt_model_zoo 의 .mxq 를 사용합니다.\n"
            "custom fetch는 --mxq <mxq> 로 로컬 경로를 지정하세요.\n"
            "custom compile은 --from-onnx <onnx> 또는 --from-pth <weights> 로 수행하세요."
        )
        raise FileNotFoundError(msg)

    if not source_description:
        source_description = f"custom/local fetch from provided .mxq: {mxq}"
    return ResolvedQBBuildRequest(
        model_or_path=str(mxq),
        source_description=source_description,
        kind=kind,
    )


def resolve_qb_build_request(*, request: QBFrontendBuildRequest) -> ResolvedQBBuildRequest:
    normalized_request = _normalize_request(request)
    normalized_request.models_dir.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_weights_export_request(normalized_request)
    if resolved is not None:
        return resolved

    resolved = _resolve_local_onnx_request(normalized_request)
    if resolved is not None:
        return resolved

    return _resolve_fetch_request(normalized_request)
