from __future__ import annotations

from pathlib import Path
from typing import Any


def unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "weights"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def prepare_supported_module_from_pth(weights_path: Path, model_name: str):
    try:
        import torch
        from torchvision import models as tv_models
    except ImportError as exc:
        raise RuntimeError(
            "torch and torchvision are required to compile from .pth/.pt weights."
        ) from exc

    normalized = model_name.lower()
    if normalized != "resnet50":
        raise ValueError(
            "--from-pth currently supports only --model-name resnet50. "
            "Use --from-onnx for other architectures."
        )

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    model = tv_models.resnet50(weights=None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        hint = ""
        if unexpected and all(key.startswith("body.") for key in unexpected[: min(len(unexpected), 10)]):
            hint = (
                " The checkpoint looks like a detector/backbone-style state_dict "
                "(e.g. keys prefixed with 'body.'), not a plain torchvision ResNet50 classifier."
            )
        raise RuntimeError(
            "Failed to load resnet50 weights cleanly from checkpoint. "
            f"missing={list(missing)}, unexpected={list(unexpected)}.{hint}"
        )
    model.eval()
    return model


def export_supported_onnx_from_pth(
    weights_path: Path,
    export_onnx_path: Path,
    model_name: str,
    input_name: str,
    input_shape: tuple[int, ...],
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to export ONNX from .pth/.pt weights.") from exc

    model = prepare_supported_module_from_pth(weights_path, model_name)
    export_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(export_onnx_path),
        input_names=[input_name],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )
    if not export_onnx_path.is_file():
        raise RuntimeError(f"ONNX export did not produce a file: {export_onnx_path}")
    return export_onnx_path
