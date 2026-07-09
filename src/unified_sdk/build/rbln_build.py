from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from unified_sdk.build.registry import register
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "vision.direct-python-compiler"
_BUILD_PIPELINE = (
    "validate_config",
    "resolve_input_info",
    "resolve_compile_options",
    "run_vendor_compile",
    "save_artifact",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "source_model": "torch.nn.Module-like object",
    "compile": "rebel.compile_from_torch(model, input_info, **compile_kwargs)",
    "save_artifact": "compiled.save(str(rbln_path))",
    "artifact": ".rbln",
}


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BuildConfig.{field_name} must be a non-empty string")
    return value.strip()


def _validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"BuildConfig.{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"BuildConfig.{field_name} must contain only positive integers: {shape!r}")
    return shape


def _validate_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    npu = extra.get("npu")
    if npu is not None and (not isinstance(npu, str) or not npu.strip()):
        raise ValueError("BuildConfig.extra['npu'] must be a non-empty string when provided")
    model_trace_method = extra.get("model_trace_method")
    if model_trace_method is not None and model_trace_method not in (
        "export",
        "export_strict",
        "jittrace",
    ):
        raise ValueError(
            "BuildConfig.extra['model_trace_method'] must be one of: "
            "'export', 'export_strict', 'jittrace'"
        )
    return extra


def _build_output_path(out_dir: str | Path, model_name: str) -> Path:
    name = _require_non_empty_string(model_name, "model_name")
    path = Path(out_dir) / name
    if path.suffix != ".rbln":
        path = path.with_suffix(".rbln")
    return path


def _capability_metadata(extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "compile_options": {
            "npu": extra.get("npu"),
            "model_trace_method": extra.get("model_trace_method"),
        },
    }


class _RBLNBuildAdapter:
    name = "rbln"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"RBLN build adapter received backend={cfg.backend!r}")

        model = cfg.model_or_path
        if not hasattr(model, "eval") or not callable(model.eval):
            raise TypeError(
                "For rbln backend, BuildConfig.model_or_path must be a torch.nn.Module-like object"
            )
        model.eval()

        if cfg.precision not in ("fp32", "fp16"):
            raise ValueError(f"Unsupported RBLN precision: {cfg.precision!r}")

        import torch
        import rebel

        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        name = _require_non_empty_string(cfg.input_name, "input_name")
        extra = _validate_extra(dict(cfg.extra or {}))
        npu = extra.get("npu")

        if cfg.bucketing_shapes:
            shapes = [
                _validate_shape(tuple(shape), f"bucketing_shapes[{idx}]")
                for idx, shape in enumerate(cfg.bucketing_shapes)
            ]
            input_info: List[Any] = [[(name, list(shape), dtype)] for shape in shapes]
        else:
            input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")
            input_info = [(name, list(input_shape), dtype)]

        compile_kwargs: Dict[str, Any] = {}
        if npu:
            compile_kwargs["npu"] = npu
        model_trace_method = extra.get("model_trace_method")
        if model_trace_method:
            compile_kwargs["model_trace_method"] = model_trace_method

        try:
            compiled = rebel.compile_from_torch(model, input_info, **compile_kwargs)
        except Exception as exc:
            raise RuntimeError(f"RBLN compile_from_torch failed: {exc}") from exc

        rbln_path = _build_output_path(cfg.out_dir, cfg.model_name)
        rbln_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            compiled.save(str(rbln_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to save RBLN model to {rbln_path}: {exc}") from exc

        meta: Dict[str, Any] = {
            "backend": self.name,
            "rbln_path": str(rbln_path),
            "input_info": input_info,
            "npu": npu,
            "precision": cfg.precision,
            "extra": extra,
            **_capability_metadata(extra),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(rbln_path),
            meta_data=meta,
        )


register(_RBLNBuildAdapter())
