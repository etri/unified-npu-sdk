from __future__ import annotations

import shutil
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
    "provided_artifact": "shutil.copyfile(src_rbln, dst_rbln)",
    "source_model": "torch.nn.Module-like object",
    "optimum_source_model": "RBLNAutoModelForImageClassification.from_pretrained(model_id, export=True, ...)",
    "onnx_restore": "onnx2torch.convert(onnx.load(path))",
    "compile": "rebel.compile_from_torch(model, input_info, **compile_kwargs)",
    "optimum_save": "compiled_optimum.save_pretrained(compiled_dir)",
    "save_artifact": "compiled.save(str(rbln_path))",
    "artifact": ".rbln",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "shutil.copyfile(src_rbln, dst_rbln)": "build_unified(cfg) for provided .rbln",
    "RBLNAutoModelForImageClassification.from_pretrained(model_id, export=True, ...)": "build_unified(cfg) for model-zoo/source fetch via optimum-rbln",
    "onnx2torch.convert(onnx.load(path))": "build_unified(cfg) for experimental ONNX -> torch restore",
    "rebel.compile_from_torch(model, input_info, **compile_kwargs)": "build_unified(cfg)",
    "compiled.save(str(rbln_path))": "BuildResult.compiled_model_path",
    ".rbln artifact": "BuildResult.meta_data['rbln_path']",
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
    compile_frontend = extra.get("compile_frontend")
    if compile_frontend is not None and compile_frontend not in (
        "rebel",
        "optimum_image_classification",
    ):
        raise ValueError(
            "BuildConfig.extra['compile_frontend'] must be one of: "
            "'rebel', 'optimum_image_classification'"
        )
    source_cache_dir = extra.get("source_cache_dir")
    if source_cache_dir is not None and not isinstance(source_cache_dir, (str, Path)):
        raise ValueError("BuildConfig.extra['source_cache_dir'] must be a string or Path when provided")
    return extra


def _build_output_path(out_dir: str | Path, model_name: str) -> Path:
    name = _require_non_empty_string(model_name, "model_name")
    path = Path(out_dir) / name
    if path.suffix != ".rbln":
        path = path.with_suffix(".rbln")
    return path


def _looks_like_path(model_or_path: Any, suffix: str) -> bool:
    return isinstance(model_or_path, (str, Path)) and str(model_or_path).lower().endswith(suffix.lower())


def _resolve_compiled_dir(path: Path) -> Path:
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


def _capability_metadata(extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "compile_options": {
            "npu": extra.get("npu"),
            "model_trace_method": extra.get("model_trace_method"),
            "compile_frontend": extra.get("compile_frontend", "rebel"),
        },
    }


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified(cfg)",
        "backend": "rbln",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


class _RBLNBuildAdapter:
    name = "rbln"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"RBLN build adapter received backend={cfg.backend!r}")

        extra = _validate_extra(dict(cfg.extra or {}))
        rbln_path = _build_output_path(cfg.out_dir, cfg.model_name)
        rbln_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(cfg.model_or_path, (str, Path)):
            candidate_path = Path(cfg.model_or_path).expanduser().resolve()
        else:
            candidate_path = None

        if _looks_like_path(cfg.model_or_path, ".rbln") or (candidate_path is not None and candidate_path.is_dir()):
            src = candidate_path
            origin_type = "provided"
            if src is None:
                raise FileNotFoundError("Provided RBLN artifact path is invalid")
            if src.is_dir():
                src = _resolve_compiled_dir(src)
                origin_type = "compiled_dir"
            if not src.is_file():
                raise FileNotFoundError(f"Provided .rbln not found: {src}")
            if src != rbln_path.resolve():
                shutil.copyfile(src, rbln_path)
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(rbln_path),
                meta_data={
                    "backend": self.name,
                    "source": origin_type,
                    "origin": str(src),
                    "rbln_path": str(rbln_path),
                    "extra": extra,
                    **_capability_metadata(extra),
                },
            )

        compile_frontend = extra.get("compile_frontend", "rebel")
        if isinstance(cfg.model_or_path, str) and compile_frontend == "optimum_image_classification":
            model_id = cfg.model_or_path.strip()
            if not model_id:
                raise ValueError("BuildConfig.model_or_path must be a non-empty model id for optimum-rbln compile")

            try:
                from optimum.rbln import RBLNAutoModelForImageClassification
            except Exception as exc:
                raise RuntimeError(
                    "optimum-rbln vision compile requires `optimum.rbln`. Install it first."
                ) from exc

            input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")
            batch_size = int(input_shape[0])
            image_size = int(input_shape[-1])
            compiled_dir = rbln_path.parent / f"{rbln_path.stem}_compiled"
            if compiled_dir.exists():
                shutil.rmtree(compiled_dir)

            compile_kwargs: Dict[str, Any] = {
                "export": True,
                "rbln_batch_size": batch_size,
                "rbln_image_size": image_size,
                "rbln_create_runtimes": False,
            }
            source_cache_dir = extra.get("source_cache_dir")
            if source_cache_dir:
                compile_kwargs["cache_dir"] = str(Path(source_cache_dir).expanduser().resolve())
            device = extra.get("device")
            if device is not None:
                compile_kwargs["rbln_device"] = device

            try:
                compiled_optimum = RBLNAutoModelForImageClassification.from_pretrained(
                    model_id,
                    **compile_kwargs,
                )
                compiled_optimum.save_pretrained(str(compiled_dir))
            except Exception as exc:
                raise RuntimeError(f"optimum-rbln image classification compile failed: {exc}") from exc

            src = _resolve_compiled_dir(compiled_dir)
            if src != rbln_path.resolve():
                shutil.copyfile(src, rbln_path)
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(rbln_path),
                meta_data={
                    "backend": self.name,
                    "source": "optimum_source_model",
                    "origin": model_id,
                    "compiled_dir": str(compiled_dir),
                    "rbln_path": str(rbln_path),
                    "extra": extra,
                    **_capability_metadata(extra),
                },
            )

        model = cfg.model_or_path
        if _looks_like_path(cfg.model_or_path, ".onnx"):
            onnx_path = Path(cfg.model_or_path).expanduser().resolve()
            if not onnx_path.is_file():
                raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
            try:
                import onnx
                from onnx2torch import convert
            except Exception as exc:
                raise RuntimeError(
                    "ONNX restore path requires `onnx` and `onnx2torch`. Install them first."
                ) from exc
            try:
                model = convert(onnx.load(str(onnx_path)))
            except Exception as exc:
                raise RuntimeError(f"Failed to restore torch model from ONNX {onnx_path}: {exc}") from exc

        if not hasattr(model, "eval") or not callable(model.eval):
            raise TypeError(
                "For rbln backend, BuildConfig.model_or_path must be a torch.nn.Module-like object, "
                "a provided .rbln path, or an experimental .onnx path."
            )
        model.eval()

        if cfg.precision not in ("fp32", "fp16"):
            raise ValueError(f"Unsupported RBLN precision: {cfg.precision!r}")

        import torch
        import rebel

        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        name = _require_non_empty_string(cfg.input_name, "input_name")
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
            hint = (
                "RBLN compile_from_torch failed. If this happens inside a CDI/container environment "
                "while host-native compile succeeds, treat it as a vendor/environment-dependent compile "
                "issue first. The currently recommended workaround is host-native compile followed by "
                "container inference/custom-fetch of the generated .rbln artifact."
            )
            raise RuntimeError(f"{hint} Original error: {exc}") from exc

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
            "source": (
                "onnx_restore"
                if _looks_like_path(cfg.model_or_path, ".onnx")
                else "torch_model"
            ),
            "extra": extra,
            **_capability_metadata(extra),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(rbln_path),
            meta_data=meta,
        )


register(_RBLNBuildAdapter())
