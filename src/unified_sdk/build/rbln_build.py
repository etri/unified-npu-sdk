from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from unified_sdk.build.registry import register
from unified_sdk.frontends import PreparedRBLNCompileSource, PreparedRBLNVisionBuildInput, ProvidedRBLNArtifact
from unified_sdk.options import RBLNVisionBuildOptions, resolve_rbln_vision_build_options
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


def _resolve_legacy_provided_artifact(cfg: BuildConfig, rbln_path: Path) -> PreparedRBLNVisionBuildInput | None:
    source_path = Path(cfg.model_or_path).expanduser().resolve() if isinstance(cfg.model_or_path, (str, Path)) else None
    if source_path is None:
        return None
    if _looks_like_path(cfg.model_or_path, ".rbln") or source_path.is_dir():
        if source_path.is_dir():
            source_path = _resolve_compiled_dir(source_path)
        return PreparedRBLNVisionBuildInput(
            kind="provided_artifact",
            provided_artifact=ProvidedRBLNArtifact(
                source_path=source_path,
                destination_path=rbln_path,
            ),
        )
    return None


def _ensure_prepared_input(cfg: BuildConfig, rbln_path: Path) -> PreparedRBLNVisionBuildInput:
    if cfg.prepared_input is not None:
        return cfg.prepared_input

    legacy_artifact = _resolve_legacy_provided_artifact(cfg, rbln_path)
    if legacy_artifact is not None:
        return legacy_artifact

    raise RuntimeError(
        "RBLN vision compile now expects a prepared frontend contract for compile sources. "
        "Call resolve_rbln_vision_build_request(...) first and pass BuildConfig(prepared_input=...). "
        "Only provided .rbln / compiled artifact directories are still accepted as a legacy direct path."
    )


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

        options = resolve_rbln_vision_build_options(
            cfg.backend_options,
            extra=dict(cfg.extra or {}),
        )
        extra = options.to_metadata()
        rbln_path = _build_output_path(cfg.out_dir, cfg.model_name)
        rbln_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_input = _ensure_prepared_input(cfg, rbln_path)

        if prepared_input.kind == "provided_artifact":
            artifact = prepared_input.provided_artifact
            if artifact is None:
                raise RuntimeError("RBLN prepared_input.kind='provided_artifact' requires provided_artifact payload")
            src = artifact.source_path
            origin_type = "provided"
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
                    "backend_options": extra,
                    **_capability_metadata(extra),
                },
            )

        compile_source = prepared_input.compile_source
        if compile_source is None:
            raise RuntimeError("RBLN prepared_input.kind='compile_source' requires compile_source payload")

        compile_frontend = compile_source.compile_frontend
        if isinstance(compile_source.source, str) and compile_frontend == "optimum_image_classification":
            model_id = compile_source.source.strip()
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
            source_cache_dir = compile_source.source_cache_dir
            if source_cache_dir:
                compile_kwargs["cache_dir"] = str(Path(source_cache_dir).expanduser().resolve())

            try:
                compiled_optimum = RBLNAutoModelForImageClassification.from_pretrained(
                    model_id,
                    **compile_kwargs,
                )
                compiled_optimum.save_pretrained(str(compiled_dir))
            except Exception as exc:
                hint = (
                    "optimum-rbln image classification compile failed. This path still uses the same "
                    "RBLN compiler backend underneath, so if host-native compile succeeds while a "
                    "CDI/container compile fails, treat it as the same vendor/environment-dependent "
                    "compile issue first. For this branch, keep the primary workflow Docker-first and "
                    "treat host-native compile only as a temporary debugging workaround while waiting "
                    "for vendor guidance."
                )
                raise RuntimeError(f"{hint} Original error: {exc}") from exc

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
                    "prepared_kind": prepared_input.kind,
                    "backend_options": extra,
                    **_capability_metadata(extra),
                },
            )

        model = compile_source.source
        if compile_source.source_label == "onnx_restore":
            onnx_path = Path(compile_source.source).expanduser().resolve()
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

        import torch
        import rebel

        dtype = torch.float16 if options.precision == "fp16" else torch.float32
        name = _require_non_empty_string(cfg.input_name, "input_name")
        npu = options.npu

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
        if options.model_trace_method:
            compile_kwargs["model_trace_method"] = options.model_trace_method

        try:
            compiled = rebel.compile_from_torch(model, input_info, **compile_kwargs)
        except Exception as exc:
            hint = (
                "RBLN compile_from_torch failed. If this happens inside a CDI/container environment "
                "while host-native compile succeeds, treat it as a vendor/environment-dependent compile "
                "issue first. For this branch, keep the primary workflow Docker-first and treat "
                "host-native compile only as a temporary debugging workaround while waiting for "
                "vendor guidance."
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
            "precision": options.precision,
            "source": compile_source.source_label,
            "prepared_kind": prepared_input.kind,
            "backend_options": extra,
            **_capability_metadata(extra),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(rbln_path),
            meta_data=meta,
        )


register(_RBLNBuildAdapter())
