from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

from .types import (
    PreparedRBLNLLMBuildInput,
    PreparedRBLNCompileSource,
    PreparedRBLNVisionBuildInput,
    ProvidedRBLNArtifact,
    RBLNLLMFrontendBuildRequest,
    RBLNVisionFrontendBuildRequest,
    ResolvedRBLNLLMBuildRequest,
    ResolvedRBLNVisionBuildRequest,
)

_MODEL_ZOO_TARGETS = {
    "resnet50": {
        "symbol": "torchvision.models.resnet50",
        "note": "official RBLN PyTorch ResNet50 model-zoo source fetch baseline (optimum-rbln path)",
        "hf_model_id": "microsoft/resnet-50",
    },
}

_LOCAL_COMPILED_DIR_PREFIXES = {"artifacts", "builds", "models", ".", ".."}


def list_model_zoo_targets() -> Dict[str, Dict[str, str]]:
    return dict(_MODEL_ZOO_TARGETS)


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def _looks_like_local_compiled_ref(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    first_part = path.parts[0] if path.parts else ""
    return path.is_absolute() or first_part in _LOCAL_COMPILED_DIR_PREFIXES


def _build_torchvision_resnet50(*, pretrained: bool):
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.eval()
    return model


def _find_weights(models_dir: Path) -> Path | None:
    candidates = sorted(models_dir.glob("resnet50*.pth")) + sorted(models_dir.glob("resnet50*.pt"))
    return candidates[0] if candidates else None


def _load_state_dict(path: Path, torch_module) -> dict:
    obj = torch_module.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"가중치 파일 형식을 해석할 수 없습니다: {path}")

    cleaned = {}
    for key, value in obj.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        if normalized.startswith("model."):
            normalized = normalized[len("model.") :]
        cleaned[normalized] = value
    return cleaned


def _restore_torch_model_from_onnx(path: Path):
    try:
        import onnx
        from onnx2torch import convert
    except Exception as exc:
        raise RuntimeError(
            "ONNX restore path requires `onnx` and `onnx2torch`. Install them first."
        ) from exc
    try:
        model = convert(onnx.load(str(path)))
    except Exception as exc:
        raise RuntimeError(f"Failed to restore torch model from ONNX {path}: {exc}") from exc
    model.eval()
    return model


def _resolve_compiled_artifact_file(path: Path) -> Path:
    candidates = sorted(path.rglob("*.rbln"))
    if not candidates:
        raise FileNotFoundError(f"No .rbln file found under compiled model directory: {path}")
    if len(candidates) > 1:
        listing = "\n".join(f"- {candidate}" for candidate in candidates)
        raise RuntimeError(
            "Multiple .rbln files were found under the compiled model directory. "
            "Please pass a single .rbln path instead.\n"
            f"{listing}"
        )
    return candidates[0]


def prepare_rbln_vision_build_input(
    model_or_path: Any,
    rbln_path: str | Path,
    *,
    source_label: str | None = None,
    provenance_kind: ResolvedRBLNVisionBuildKind | None = None,
    provenance_detail: str | None = None,
    compile_frontend: Literal["rebel", "optimum_image_classification"] | None = None,
    source_cache_dir: Path | None = None,
    source_path: Path | None = None,
) -> PreparedRBLNVisionBuildInput:
    destination = Path(rbln_path).expanduser().resolve()
    resolved_source_path = Path(model_or_path).expanduser().resolve() if isinstance(model_or_path, (str, Path)) else None

    if resolved_source_path is not None and (resolved_source_path.suffix == ".rbln" or resolved_source_path.is_dir()):
        artifact_path = resolved_source_path
        if artifact_path.is_dir():
            artifact_path = _resolve_compiled_artifact_file(artifact_path)
        return PreparedRBLNVisionBuildInput(
            kind="provided_artifact",
            provided_artifact=ProvidedRBLNArtifact(
                source_path=artifact_path,
                destination_path=destination,
            ),
        )

    normalized_compile_frontend = compile_frontend or "rebel"
    normalized_source_label = source_label or "torch_model"
    normalized_provenance_kind = provenance_kind or "torch_model"
    normalized_provenance_detail = provenance_detail or normalized_source_label
    source = model_or_path
    if resolved_source_path is not None and resolved_source_path.suffix == ".onnx":
        source = _restore_torch_model_from_onnx(resolved_source_path)
    elif isinstance(model_or_path, str):
        normalized_source_label = source_label or "optimum_source_model"
        normalized_compile_frontend = compile_frontend or "optimum_image_classification"
        normalized_provenance_kind = provenance_kind or "optimum_source_model"
        normalized_provenance_detail = provenance_detail or str(model_or_path)

    return PreparedRBLNVisionBuildInput(
        kind="compile_source",
        compile_source=PreparedRBLNCompileSource(
            source=source,
            source_label=normalized_source_label,
            provenance_kind=normalized_provenance_kind,
            provenance_detail=normalized_provenance_detail,
            source_path=source_path,
            compile_frontend=normalized_compile_frontend,
            source_cache_dir=source_cache_dir,
        ),
    )


def resolve_rbln_llm_build_request(request: RBLNLLMFrontendBuildRequest) -> ResolvedRBLNLLMBuildRequest:
    model_ref = str(request.model_ref).strip()
    if not model_ref:
        raise ValueError("RBLNLLMFrontendBuildRequest.model_ref must be a non-empty string or path")

    build_mode = request.build_mode
    if build_mode == "fetch":
        return ResolvedRBLNLLMBuildRequest(
            source_description=f"runtime model-ref passthrough: {model_ref}",
            kind="runtime_model_ref",
            prepared_input=PreparedRBLNLLMBuildInput(
                kind="runtime_model_ref",
                model_ref=model_ref,
                artifact_dir=None,
            ),
        )

    artifact_dir = request.out_dir.expanduser().resolve() / request.model_name.strip()
    return ResolvedRBLNLLMBuildRequest(
        source_description=f"artifact build from model ref via optimum-rbln: {model_ref}",
        kind="artifact_build",
        prepared_input=PreparedRBLNLLMBuildInput(
            kind="artifact_build",
            model_ref=model_ref,
            artifact_dir=artifact_dir,
        ),
    )


def resolve_rbln_vision_build_request(request: RBLNVisionFrontendBuildRequest) -> ResolvedRBLNVisionBuildRequest:
    models_dir = request.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    model_name = request.model_name.strip()
    if not model_name:
        raise ValueError("RBLNVisionFrontendBuildRequest.model_name must be a non-empty string")

    compiled_model_ref = (request.compiled_model_ref or "").strip()
    if compiled_model_ref:
        ref_path = Path(compiled_model_ref).expanduser()
        if ref_path.exists():
            resolved = ref_path.resolve()
            return ResolvedRBLNVisionBuildRequest(
                model_or_path=str(resolved),
                source_description=f"standard fetch from local compiled RBLN directory: {resolved}",
                kind="compiled_dir",
                prepared_input=prepare_rbln_vision_build_input(
                    str(resolved),
                    models_dir / f"{model_name}.rbln",
                ),
            )
        if _looks_like_local_compiled_ref(compiled_model_ref):
            raise FileNotFoundError(
                "compiled-model-ref was interpreted as a local compiled RBLN directory, "
                f"but it does not exist: {ref_path.resolve()}\n"
                "If you intended a Hugging Face repo id, pass an explicit repo id like "
                "'org/repo-name'. If you intended a local directory, ensure it contains "
                "*.rbln and rbln_config.json."
            )
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "compiled-model-ref hub fetch requires `huggingface_hub`. Install it first."
            ) from exc
        local_dir = models_dir / compiled_model_ref.split("/")[-1]
        snapshot_path = Path(
            snapshot_download(
                repo_id=compiled_model_ref,
                local_dir=str(local_dir),
            )
        ).resolve()
        return ResolvedRBLNVisionBuildRequest(
            model_or_path=str(snapshot_path),
            source_description=f"standard fetch from compiled RBLN artifact repo: {compiled_model_ref} -> {snapshot_path}",
            kind="compiled_dir",
            prepared_input=prepare_rbln_vision_build_input(
                str(snapshot_path),
                models_dir / f"{model_name}.rbln",
            ),
        )

    if request.provided_rbln is not None:
        provided = request.provided_rbln.expanduser().resolve()
        return ResolvedRBLNVisionBuildRequest(
            model_or_path=str(provided),
            source_description=f"provided .rbln fetch: {provided}",
            kind="provided_artifact",
            prepared_input=prepare_rbln_vision_build_input(
                str(provided),
                models_dir / f"{model_name}.rbln",
            ),
        )

    if request.from_onnx is not None:
        onnx_path = request.from_onnx.expanduser().resolve()
        model = _restore_torch_model_from_onnx(onnx_path)
        return ResolvedRBLNVisionBuildRequest(
            model_or_path=model,
            source_description=f"experimental/unverified ONNX restore -> .rbln: {onnx_path}",
            kind="onnx_restore",
            prepared_input=prepare_rbln_vision_build_input(
                model,
                models_dir / f"{model_name}.rbln",
                source_label="onnx_restored_torch_model",
                provenance_kind="onnx_restore",
                provenance_detail=f"experimental/unverified ONNX restore: {onnx_path}",
                compile_frontend="rebel",
                source_path=onnx_path,
            ),
        )

    model_zoo_target = _normalize_model_name(request.model_zoo_model or "")
    if model_zoo_target:
        if model_zoo_target not in _MODEL_ZOO_TARGETS:
            raise ValueError(
                f"Unsupported model-zoo target: {request.model_zoo_model!r}. "
                f"Try one of: {', '.join(sorted(_MODEL_ZOO_TARGETS))}"
            )
        if request.pretrained:
            model_ref = _MODEL_ZOO_TARGETS[model_zoo_target]["hf_model_id"]
            prepared = prepare_rbln_vision_build_input(
                model_ref,
                models_dir / f"{model_name}.rbln",
                source_label="optimum_source_model",
                provenance_kind="optimum_source_model",
                provenance_detail=f"standard model-zoo/source hub fetch: {model_ref}",
                compile_frontend="optimum_image_classification",
                source_cache_dir=(models_dir / "hf-cache").resolve(),
            )
            return ResolvedRBLNVisionBuildRequest(
                model_or_path=model_ref,
                source_description=(
                    "standard fetch from model-zoo/source hub via optimum-rbln -> .rbln: "
                    f"{model_ref}"
                ),
                kind="optimum_source_model",
                prepared_input=prepared,
            )

        model = _build_torchvision_resnet50(pretrained=False)
        return ResolvedRBLNVisionBuildRequest(
            model_or_path=model,
            source_description=(
                "reference compile from official RBLN model-zoo/tutorial baseline: "
                "torchvision ResNet50 local/random-init"
            ),
            kind="torch_model",
            prepared_input=prepare_rbln_vision_build_input(
                model,
                models_dir / f"{model_name}.rbln",
                source_label="torchvision_resnet50_random_init",
                provenance_kind="torch_model",
                provenance_detail="reference compile from torchvision ResNet50 random-init",
                compile_frontend="rebel",
            ),
        )

    weights_path = request.weights_path.expanduser().resolve() if request.weights_path is not None else _find_weights(models_dir)
    if request.require_weights and weights_path is None:
        raise FileNotFoundError(
            f"{models_dir} 에서 resnet50 가중치 파일을 찾지 못했습니다.\n"
            f"예) {models_dir/'resnet50.pth'} 또는 {models_dir/'resnet50_state_dict.pth'}"
        )

    import torch

    model = _build_torchvision_resnet50(pretrained=False)
    if weights_path is not None:
        state_dict = _load_state_dict(weights_path, torch)
        model.load_state_dict(state_dict, strict=False)
        source_description = f"user PTH/PT -> torch restore -> .rbln: {weights_path}"
        resolved_kind: ResolvedRBLNVisionBuildKind = "pth_restore"
    else:
        source_description = "local torchvision ResNet50 random-init -> .rbln"
        resolved_kind = "torch_model"

    return ResolvedRBLNVisionBuildRequest(
        model_or_path=model,
        source_description=source_description,
        kind=resolved_kind,
        prepared_input=prepare_rbln_vision_build_input(
            model,
            models_dir / f"{model_name}.rbln",
            source_label="pth_restored_torch_model" if weights_path is not None else "torch_model",
            provenance_kind=resolved_kind,
            provenance_detail=source_description,
            compile_frontend="rebel",
            source_path=weights_path,
        ),
    )


__all__ = [
    "PreparedRBLNLLMBuildInput",
    "PreparedRBLNCompileSource",
    "PreparedRBLNVisionBuildInput",
    "ProvidedRBLNArtifact",
    "RBLNLLMFrontendBuildRequest",
    "RBLNVisionFrontendBuildRequest",
    "ResolvedRBLNLLMBuildRequest",
    "ResolvedRBLNVisionBuildRequest",
    "list_model_zoo_targets",
    "resolve_rbln_llm_build_request",
    "prepare_rbln_vision_build_input",
    "resolve_rbln_vision_build_request",
]
