from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.frontends.prepare_warboy_source import prepare_warboy_build_input
from unified_sdk.frontends.types import ResolvedWarboyBuildRequest, WarboyFrontendBuildRequest
from unified_sdk.frontends.warboy_model_zoo import fetch_model_zoo_enf, find_local_enf


_FRONTEND_PIPELINE = (
    "normalize_request",
    "resolve_compile_source_or_artifact",
    "resolve_local_or_model_zoo_enf",
    "emit_resolved_build_request",
)
_FRONTEND_API_MAP = {
    "provided_enf": "resolve_warboy_build_request(request=WarboyFrontendBuildRequest(provided_enf=...))",
    "local_enf": "frontends.find_local_enf(models_dir, model_name)",
    "model_zoo_enf": "frontends.fetch_model_zoo_enf(model_name, target_npu, models_dir)",
    "quantized_onnx": "resolve_warboy_build_request(request=WarboyFrontendBuildRequest(from_onnx=...))",
}


def describe_frontend_api_mapping() -> Dict[str, Any]:
    return {
        "unified_frontend_api": "resolve_warboy_build_request(request=WarboyFrontendBuildRequest(...))",
        "capability_family": "vision.frontend-prepare-fetch",
        "pipeline": _FRONTEND_PIPELINE,
        "vendor_api_map": _FRONTEND_API_MAP,
    }


def resolve_warboy_build_request(*, request: WarboyFrontendBuildRequest) -> ResolvedWarboyBuildRequest:
    models_dir = request.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if request.from_onnx is not None:
        onnx_path = request.from_onnx.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"quantized ONNX not found: {onnx_path}")
        return ResolvedWarboyBuildRequest(
            model_or_path=str(onnx_path),
            source_description=f"furiosa-compiler from quantized ONNX: {onnx_path}",
            kind="quantized_onnx",
            prepared_input=prepare_warboy_build_input(str(onnx_path), models_dir / f"{request.model_name}.enf"),
        )

    fetched_from_model_zoo = False
    enf = request.provided_enf.expanduser().resolve() if request.provided_enf else find_local_enf(models_dir, request.model_name)
    if enf is None:
        enf = fetch_model_zoo_enf(request.model_name, request.target_npu, models_dir)
        fetched_from_model_zoo = enf is not None

    if enf is None:
        msg = (
            f"{models_dir} 에서 {request.model_name}*.enf 를 찾지 못했고, "
            "Furiosa model zoo 에서도 대응 ENF 를 확보하지 못했습니다.\n"
            "사전 컴파일된 .enf 를 제공하거나, quantized ONNX 를 prepare한 뒤 빌드하세요."
        )
        raise FileNotFoundError(msg)

    if fetched_from_model_zoo:
        source_description = f"standard fetch from Furiosa model zoo ENF: {enf}"
        kind = "model_zoo_enf"
    elif request.provided_enf is not None:
        source_description = f"provided .enf: {enf}"
        kind = "provided_artifact"
    else:
        source_description = f"local .enf from models/: {enf}"
        kind = "local_enf"

    return ResolvedWarboyBuildRequest(
        model_or_path=str(enf),
        source_description=source_description,
        kind=kind,
        prepared_input=prepare_warboy_build_input(str(enf), models_dir / f"{request.model_name}.enf"),
    )
