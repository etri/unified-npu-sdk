from __future__ import annotations

from pathlib import Path

from unified_sdk.frontends.export_qb_onnx import export_supported_onnx_from_pth
from unified_sdk.frontends.qb_model_zoo import (
    find_local_mxq,
    find_model_zoo_mxq,
    normalize_mxq_into_models,
    trigger_model_zoo_fetch,
)
from unified_sdk.frontends.types import ResolvedQBBuildRequest


def resolve_qb_build_request(
    *,
    model_name: str,
    models_dir: Path,
    product: str,
    core_mode: str,
    from_pth: Path | None = None,
    from_onnx: Path | None = None,
    provided_mxq: Path | None = None,
    export_onnx_path: Path | None = None,
    input_name: str = "input",
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    require_mxq: bool = False,
) -> ResolvedQBBuildRequest:
    models_dir = models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if from_pth is not None:
        weights_path = from_pth.expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"PTH/PT weights not found: {weights_path}")
        resolved_export_path = (
            export_onnx_path.expanduser().resolve()
            if export_onnx_path is not None
            else (models_dir / f"{model_name}.onnx").resolve()
        )
        onnx_path = export_supported_onnx_from_pth(
            weights_path=weights_path,
            export_onnx_path=resolved_export_path,
            model_name=model_name,
            input_name=input_name,
            input_shape=input_shape,
        )
        return ResolvedQBBuildRequest(
            model_or_path=str(onnx_path),
            source_description=f"local weights -> ONNX export -> compiler Python API compile: {weights_path} -> {onnx_path}",
        )

    if from_onnx is not None:
        onnx_path = from_onnx.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        return ResolvedQBBuildRequest(
            model_or_path=str(onnx_path),
            source_description=f"local/custom ONNX -> compiler Python API compile: {onnx_path}",
        )

    mxq = provided_mxq.expanduser().resolve() if provided_mxq is not None else find_local_mxq(models_dir, model_name)
    source_description = ""
    if mxq is None:
        mxq = find_model_zoo_mxq(model_name, product, core_mode)
        if mxq is None:
            mxq = trigger_model_zoo_fetch(model_name, product, core_mode, models_dir)
        if mxq is not None:
            normalized_mxq = normalize_mxq_into_models(mxq, models_dir, model_name)
            source_description = f"standard fetch from official model zoo: {mxq} -> {normalized_mxq}"
            mxq = normalized_mxq

    if mxq is None:
        msg = (
            f"{models_dir} 또는 ~/.mblt_model_zoo/vision/{product}/{core_mode} 에서 "
            f"{model_name}*.mxq 를 찾지 못했습니다.\n"
            "표준 fetch는 ~/.mblt_model_zoo 의 .mxq 를 사용합니다.\n"
            "custom fetch는 --mxq <mxq> 로 로컬 경로를 지정하세요.\n"
            "custom compile은 --from-onnx <onnx> 또는 --from-pth <weights> 로 수행하세요."
        )
        if require_mxq:
            raise FileNotFoundError(msg)
        raise FileNotFoundError(msg)

    if not source_description:
        source_description = f"custom/local fetch from provided .mxq: {mxq}"
    return ResolvedQBBuildRequest(
        model_or_path=str(mxq),
        source_description=source_description,
    )
