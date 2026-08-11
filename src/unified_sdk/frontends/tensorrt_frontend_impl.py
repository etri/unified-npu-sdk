from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from .types import (
    PreparedTensorRTCompileSource,
    PreparedTensorRTLLMFetchInput,
    PreparedTensorRTLLMBuildInput,
    PreparedTensorRTVisionBuildInput,
    ProvidedTensorRTArtifact,
    ResolvedTensorRTLLMFetchRequest,
    ResolvedTensorRTLLMBuildRequest,
    TensorRTLLMFrontendFetchRequest,
    ResolvedTensorRTVisionBuildRequest,
    TensorRTLLMFrontendBuildRequest,
    TensorRTVisionFrontendBuildRequest,
)


_LOCAL_ENGINE_PREFIXES = {"artifacts", "build_output", "models", ".", ".."}
_TRTLLM_ARTIFACT_MARKERS = ("executor_config.json", "engine_config.json")


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def list_torchvision_model_zoo_targets() -> list[str]:
    try:
        from torchvision import models as tv_models
    except Exception:
        return []

    if hasattr(tv_models, "list_models"):
        try:
            return sorted(str(name) for name in tv_models.list_models())
        except Exception:
            pass

    names: list[str] = []
    for name in dir(tv_models):
        if name.startswith("_"):
            continue
        candidate = getattr(tv_models, name, None)
        if callable(candidate) and name.lower() == name:
            names.append(name)
    return sorted(set(names))


def _resolve_torchvision_model_name(model_name: str) -> str | None:
    normalized = _normalize_model_name(model_name)
    for candidate in list_torchvision_model_zoo_targets():
        if _normalize_model_name(candidate) == normalized:
            return candidate
    return None


def _find_onnx(models_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*.onnx")) + sorted(models_dir.glob("*.onnx"))
    return candidates[0].resolve() if candidates else None


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "weights", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def _strip_known_prefixes(key: str) -> str:
    while True:
        updated = key
        for prefix in ("module.", "model.", "net.", "network."):
            if updated.startswith(prefix):
                updated = updated[len(prefix) :]
        if updated == key:
            return key
        key = updated


def _score_prefix_strip(state_dict: dict[str, Any], prefix: str, expected_keys: set[str]) -> tuple[int, dict[str, Any]]:
    plen = len(prefix)
    transformed = {}
    hits = 0
    for key, value in state_dict.items():
        stripped = key[plen:] if key.startswith(prefix) else key
        transformed[stripped] = value
        if stripped in expected_keys:
            hits += 1
    return hits, transformed


def _align_state_dict_namespaces(state_dict: dict[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    cleaned = {_strip_known_prefixes(k): v for k, v in state_dict.items()}
    base_hits = sum(1 for k in cleaned if k in expected_keys)
    best_hits = base_hits
    best = cleaned

    first_segments = sorted({k.split(".", 1)[0] for k in cleaned if "." in k})
    for seg in first_segments:
        prefix = seg + "."
        hits, transformed = _score_prefix_strip(cleaned, prefix, expected_keys)
        if hits > best_hits:
            best_hits = hits
            best = transformed
    return best


def _resolve_torchvision_model(model_name: str, *, pretrained: bool):
    try:
        from torchvision import models as tv_models
    except ImportError as exc:
        raise RuntimeError("torchvision is required for torchvision model zoo fetching and .pth export.") from exc

    resolved_name = _resolve_torchvision_model_name(model_name)
    if resolved_name is None:
        raise ValueError(
            f"Unsupported torchvision model name: {model_name!r}. "
            "Use list_torchvision_model_zoo_targets() to inspect available standard fetch targets."
        )

    if hasattr(tv_models, "get_model"):
        kwargs = {}
        if pretrained:
            if hasattr(tv_models, "get_model_weights"):
                weights_enum = tv_models.get_model_weights(resolved_name)
                kwargs["weights"] = weights_enum.DEFAULT
            else:
                kwargs["pretrained"] = True
        else:
            if hasattr(tv_models, "get_model_weights"):
                kwargs["weights"] = None
            else:
                kwargs["pretrained"] = False
        return resolved_name, tv_models.get_model(resolved_name, **kwargs)

    factory = getattr(tv_models, resolved_name, None)
    if not callable(factory):
        raise ValueError(f"Resolved torchvision model is not callable: {resolved_name}")
    return resolved_name, factory(pretrained=pretrained)


def _prepare_module_from_pth(weights_path: Path, model_name: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to compile from .pth/.pt weights.") from exc

    resolved_name, model = _resolve_torchvision_model(model_name, pretrained=False)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    aligned = _align_state_dict_namespaces(state_dict, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(aligned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load {resolved_name} weights cleanly from checkpoint. "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    return resolved_name, model


def _export_module_to_onnx(model, export_onnx_path: Path, input_name: str, input_shape: tuple[int, ...]) -> Path:
    try:
        import inspect
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to export ONNX.") from exc

    export_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(input_shape, dtype=torch.float32)
    export_kwargs = {
        "input_names": [input_name],
        "output_names": ["output"],
        "opset_version": 13,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(model, dummy, str(export_onnx_path), **export_kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        export_kwargs.pop("dynamo", None)
        torch.onnx.export(model, dummy, str(export_onnx_path), **export_kwargs)
    if not export_onnx_path.is_file():
        raise RuntimeError(f"ONNX export did not produce a file: {export_onnx_path}")
    return export_onnx_path.resolve()


def _looks_like_local_engine_ref(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    first_part = path.parts[0] if path.parts else ""
    return path.is_absolute() or first_part in _LOCAL_ENGINE_PREFIXES


def _looks_like_local_llm_ref(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    first_part = path.parts[0] if path.parts else ""
    return path.is_absolute() or first_part in _LOCAL_ENGINE_PREFIXES


def _detect_trtllm_artifact_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(path.glob("*.engine")):
        return True
    return any((path / marker).exists() for marker in _TRTLLM_ARTIFACT_MARKERS)


def _detect_trtllm_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    patterns = ("rank*.safetensors", "rank*.bin", "rank*.pt", "rank*.ckpt")
    return any(any(path.glob(pattern)) for pattern in patterns)


def classify_tensorrt_llm_source(model_ref: str | Path) -> tuple[str, Path | None]:
    model_ref_str = str(model_ref).strip()
    local_path = Path(model_ref_str).expanduser()
    source_kind = "model_id"
    source_path = None
    if local_path.exists():
        source_path = local_path.resolve()
        if _detect_trtllm_checkpoint_dir(source_path):
            source_kind = "local_checkpoint_dir"
        elif _detect_trtllm_artifact_dir(source_path):
            source_kind = "local_artifact_dir"
        else:
            source_kind = "local_model_path"
    elif _looks_like_local_llm_ref(model_ref_str):
        raise FileNotFoundError(
            f"TensorRT-LLM local path was requested but does not exist: {local_path}. "
            "If you intended a Hugging Face repo id, pass an explicit repo id like 'org/model'."
        )
    return source_kind, source_path


def _build_output_engine_path(out_dir: Path, model_name: str, precision: str) -> Path:
    return (out_dir / f"{model_name}_{precision.upper()}.engine").resolve()


def prepare_tensorrt_vision_build_input(
    model_or_path: str | Path,
    engine_path: str | Path,
    *,
    source_label: str,
    provenance_kind,
    provenance_detail: str,
    input_name: str,
    min_input_shape: tuple[int, ...],
    opt_input_shape: tuple[int, ...],
    max_input_shape: tuple[int, ...],
) -> PreparedTensorRTVisionBuildInput:
    destination = Path(engine_path).expanduser().resolve()
    source_path = Path(model_or_path).expanduser().resolve()

    if source_path.suffix.lower() == ".engine":
        return PreparedTensorRTVisionBuildInput(
            kind="provided_artifact",
            provided_artifact=ProvidedTensorRTArtifact(
                source_path=source_path,
                destination_path=destination,
            ),
        )

    return PreparedTensorRTVisionBuildInput(
        kind="compile_source",
        compile_source=PreparedTensorRTCompileSource(
            source_path=source_path,
            source_label=source_label,
            provenance_kind=provenance_kind,
            provenance_detail=provenance_detail,
            input_name=input_name,
            min_input_shape=min_input_shape,
            opt_input_shape=opt_input_shape,
            max_input_shape=max_input_shape,
        ),
    )


def resolve_tensorrt_vision_build_request(request: TensorRTVisionFrontendBuildRequest) -> ResolvedTensorRTVisionBuildRequest:
    models_dir = request.models_dir.expanduser().resolve()
    out_dir = request.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = request.model_name.strip()
    if not model_name:
        raise ValueError("TensorRTVisionFrontendBuildRequest.model_name must be a non-empty string")

    engine_path = _build_output_engine_path(out_dir, model_name, request.precision)
    input_name = request.input_name.strip()
    if not input_name:
        raise ValueError("TensorRTVisionFrontendBuildRequest.input_name must be a non-empty string")

    if request.provided_engine is not None:
        engine = request.provided_engine.expanduser().resolve()
        if not engine.is_file():
            raise FileNotFoundError(f"Engine not found: {engine}")
        return ResolvedTensorRTVisionBuildRequest(
            model_or_path=str(engine),
            source_description=f"custom/local fetch from provided .engine: {engine}",
            kind="provided_artifact",
            prepared_input=prepare_tensorrt_vision_build_input(
                engine,
                engine_path,
                source_label="provided_engine",
                provenance_kind="provided_artifact",
                provenance_detail=f"provided .engine fetch: {engine}",
                input_name=input_name,
                min_input_shape=request.min_input_shape,
                opt_input_shape=request.opt_input_shape,
                max_input_shape=request.max_input_shape,
            ),
        )

    if request.onnx_path is not None:
        onnx_path = request.onnx_path.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        return ResolvedTensorRTVisionBuildRequest(
            model_or_path=str(onnx_path),
            source_description=f"local/custom ONNX -> TensorRT compile: {onnx_path}",
            kind="onnx_path",
            prepared_input=prepare_tensorrt_vision_build_input(
                onnx_path,
                engine_path,
                source_label="onnx_path",
                provenance_kind="onnx_path",
                provenance_detail=f"local/custom ONNX -> TensorRT compile: {onnx_path}",
                input_name=input_name,
                min_input_shape=request.min_input_shape,
                opt_input_shape=request.opt_input_shape,
                max_input_shape=request.max_input_shape,
            ),
        )

    if request.weights_path is not None:
        weights_path = request.weights_path.expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"PTH/PT weights not found: {weights_path}")
        resolved_name, model = _prepare_module_from_pth(weights_path, model_name)
        export_onnx_path = (
            request.export_onnx_path.expanduser().resolve()
            if request.export_onnx_path is not None
            else (models_dir / f"{model_name}.onnx").resolve()
        )
        onnx_path = _export_module_to_onnx(
            model=model,
            export_onnx_path=export_onnx_path,
            input_name=input_name,
            input_shape=request.opt_input_shape,
        )
        detail = f"local weights -> ONNX export -> TensorRT compile: {weights_path} -> {onnx_path} ({resolved_name})"
        return ResolvedTensorRTVisionBuildRequest(
            model_or_path=str(onnx_path),
            source_description=detail,
            kind="pth_export",
            prepared_input=prepare_tensorrt_vision_build_input(
                onnx_path,
                engine_path,
                source_label="pth_export",
                provenance_kind="pth_export",
                provenance_detail=detail,
                input_name=input_name,
                min_input_shape=request.min_input_shape,
                opt_input_shape=request.opt_input_shape,
                max_input_shape=request.max_input_shape,
            ),
        )

    auto_onnx = _find_onnx(models_dir, model_name)
    if auto_onnx is not None:
        return ResolvedTensorRTVisionBuildRequest(
            model_or_path=str(auto_onnx),
            source_description=f"local/custom ONNX -> TensorRT compile: {auto_onnx}",
            kind="onnx_path",
            prepared_input=prepare_tensorrt_vision_build_input(
                auto_onnx,
                engine_path,
                source_label="onnx_path",
                provenance_kind="onnx_path",
                provenance_detail=f"local/custom ONNX -> TensorRT compile: {auto_onnx}",
                input_name=input_name,
                min_input_shape=request.min_input_shape,
                opt_input_shape=request.opt_input_shape,
                max_input_shape=request.max_input_shape,
            ),
        )

    if request.model_zoo_model:
        resolved_name, model = _resolve_torchvision_model(request.model_zoo_model, pretrained=request.pretrained)
        export_onnx_path = (
            request.export_onnx_path.expanduser().resolve()
            if request.export_onnx_path is not None
            else (models_dir / f"{model_name}.onnx").resolve()
        )
        onnx_path = _export_module_to_onnx(
            model=model,
            export_onnx_path=export_onnx_path,
            input_name=input_name,
            input_shape=request.opt_input_shape,
        )
        detail = (
            "standard fetch from torchvision model zoo -> ONNX export -> TensorRT compile: "
            f"{resolved_name} -> {onnx_path}"
        )
        return ResolvedTensorRTVisionBuildRequest(
            model_or_path=str(onnx_path),
            source_description=detail,
            kind="torchvision_export",
            prepared_input=prepare_tensorrt_vision_build_input(
                onnx_path,
                engine_path,
                source_label="torchvision_export",
                provenance_kind="torchvision_export",
                provenance_detail=detail,
                input_name=input_name,
                min_input_shape=request.min_input_shape,
                opt_input_shape=request.opt_input_shape,
                max_input_shape=request.max_input_shape,
            ),
        )

    if request.require_onnx:
        raise FileNotFoundError(
            f"{models_dir} 에서 {model_name} ONNX 를 찾지 못했습니다.\n"
            f"예) {models_dir / (model_name + '.onnx')}"
        )

    resolved_name, model = _resolve_torchvision_model(model_name, pretrained=True)
    export_onnx_path = (
        request.export_onnx_path.expanduser().resolve()
        if request.export_onnx_path is not None
        else (models_dir / f"{model_name}.onnx").resolve()
    )
    onnx_path = _export_module_to_onnx(
        model=model,
        export_onnx_path=export_onnx_path,
        input_name=input_name,
        input_shape=request.opt_input_shape,
    )
    detail = f"standard fetch from torchvision model zoo -> ONNX export -> TensorRT compile: {resolved_name} -> {onnx_path}"
    return ResolvedTensorRTVisionBuildRequest(
        model_or_path=str(onnx_path),
        source_description=detail,
        kind="torchvision_export",
        prepared_input=prepare_tensorrt_vision_build_input(
            onnx_path,
            engine_path,
            source_label="torchvision_export",
            provenance_kind="torchvision_export",
            provenance_detail=detail,
            input_name=input_name,
            min_input_shape=request.min_input_shape,
            opt_input_shape=request.opt_input_shape,
            max_input_shape=request.max_input_shape,
        ),
    )


def resolve_tensorrt_llm_fetch_request(request: TensorRTLLMFrontendFetchRequest) -> ResolvedTensorRTLLMFetchRequest:
    model_ref = str(request.model_ref).strip()
    if not model_ref:
        raise ValueError("TensorRTLLMFrontendFetchRequest.model_ref must be a non-empty string or path")

    source_kind, source_path = classify_tensorrt_llm_source(model_ref)

    if source_kind == "local_checkpoint_dir":
        raise ValueError(
            "TensorRT-LLM checkpoint dir is a custom compile input, not a runtime fetch input. "
            "Use resolve_tensorrt_llm_build_request(...) for checkpoint-dir compile."
        )
    if source_kind == "model_id":
        description = f"runtime model-id fetch passthrough: {model_ref}"
    elif source_kind == "local_artifact_dir":
        description = f"runtime local prebuilt TensorRT-LLM artifact dir passthrough: {source_path}"
    else:
        description = f"runtime local model path passthrough: {source_path}"
    return ResolvedTensorRTLLMFetchRequest(
        source_description=description,
        kind="runtime_model_ref",
        prepared_input=PreparedTensorRTLLMFetchInput(
            kind="runtime_model_ref",
            model_ref=model_ref,
            source_kind=source_kind,
            source_path=source_path,
        ),
    )


def resolve_tensorrt_llm_build_request(request: TensorRTLLMFrontendBuildRequest) -> ResolvedTensorRTLLMBuildRequest:
    model_ref = str(request.model_ref).strip()
    if not model_ref:
        raise ValueError("TensorRTLLMFrontendBuildRequest.model_ref must be a non-empty string or path")

    source_kind, source_path = classify_tensorrt_llm_source(model_ref)

    artifact_dir = request.out_dir.expanduser().resolve() / request.model_name.strip()
    if source_kind == "local_artifact_dir":
        raise ValueError(
            "A prebuilt TensorRT-LLM artifact directory is a runtime fetch input, not a compile input. "
            "Use resolve_tensorrt_llm_fetch_request(...) for local artifact/runtime passthrough."
        )
    compile_variant = "checkpoint_dir_cli" if source_kind == "local_checkpoint_dir" else "model_ref_api"
    if compile_variant == "checkpoint_dir_cli":
        description = f"TensorRT-LLM custom compile from local checkpoint dir via trtllm-build: {source_path}"
    elif source_kind == "model_id":
        description = f"TensorRT-LLM custom compile from model id via Python API: {model_ref}"
    else:
        description = f"TensorRT-LLM custom compile from local model path via Python API: {source_path}"
    return ResolvedTensorRTLLMBuildRequest(
        source_description=description,
        kind="artifact_build",
        prepared_input=PreparedTensorRTLLMBuildInput(
            kind="artifact_build",
            model_ref=model_ref,
            source_kind=source_kind,
            source_path=source_path,
            artifact_dir=artifact_dir,
            compile_variant=compile_variant,
            checkpoint_dir=source_path if compile_variant == "checkpoint_dir_cli" else None,
        ),
    )


__all__ = [
    "PreparedTensorRTCompileSource",
    "PreparedTensorRTLLMFetchInput",
    "PreparedTensorRTLLMBuildInput",
    "PreparedTensorRTVisionBuildInput",
    "ProvidedTensorRTArtifact",
    "ResolvedTensorRTLLMFetchRequest",
    "ResolvedTensorRTLLMBuildRequest",
    "TensorRTLLMFrontendFetchRequest",
    "ResolvedTensorRTVisionBuildRequest",
    "TensorRTLLMFrontendBuildRequest",
    "TensorRTVisionFrontendBuildRequest",
    "classify_tensorrt_llm_source",
    "list_torchvision_model_zoo_targets",
    "prepare_tensorrt_vision_build_input",
    "resolve_tensorrt_llm_fetch_request",
    "resolve_tensorrt_llm_build_request",
    "resolve_tensorrt_vision_build_request",
]
