from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

from unified_sdk.build.registry import register
from unified_sdk.frontends import place_provided_qb_artifact, prepare_qb_build_input
from unified_sdk.options import QBBuildOptions, resolve_qb_build_options
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "vision.direct-python-compiler"
_BUILD_PIPELINE = (
    "validate_config",
    "resolve_artifact_or_source",
    "resolve_compile_options",
    "run_vendor_compile_or_place_artifact",
    "verify_artifact",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "provided_artifact": "frontends.place_provided_qb_artifact(provided_mxq, mxq_path)",
    "prepare": "frontends.prepare_qb_build_input(model_or_path, mxq_path)",
    "compile": "compiler_python_api.mxq_compile(**compile_kwargs)",
    "calibration": "QBBuildOptions.calib_data_path or use_random_calib",
    "artifact": ".mxq",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "frontends.place_provided_qb_artifact(provided_mxq, mxq_path)": "build_unified(cfg) for provided .mxq",
    "frontends.prepare_qb_build_input(model_or_path, mxq_path)": "build_unified(cfg) prepare/fetch step",
    "compiler_python_api.mxq_compile(**compile_kwargs)": "build_unified(cfg) for ONNX/torch compile",
    "QBBuildOptions.calib_data_path or use_random_calib": "QBBuildOptions.calib_data_path / use_random_calib",
    ".mxq artifact": "BuildResult.compiled_model_path",
}


def _resolve_mxq_compile():
    """Resolve the Mobilint compiler API regardless of package exposure name.

    Vendor wheel filenames may be `qbcompiler-*.whl`, while the Python import
    is still exposed as `qubee` on some versions. Newer wheels may expose the
    top-level package as `qbcompiler`. We support both here.
    """
    errors: list[str] = []
    for module_name in ("qubee", "qbcompiler"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        mxq_compile = getattr(module, "mxq_compile", None)
        if callable(mxq_compile):
            return module_name, mxq_compile
        errors.append(f"{module_name}: missing mxq_compile")
    raise RuntimeError(
        "Mobilint compiler Python API is required to compile a model to .mxq. "
        "Expected a vendor wheel exposing `qubee.mxq_compile(...)` or "
        "`qbcompiler.mxq_compile(...)`. "
        f"Checked modules: {', '.join(errors) or 'none'}"
    )


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
    if name.lower().endswith(".mxq"):
        return Path(out_dir) / name
    return Path(out_dir) / f"{name}.mxq"


def _capability_metadata(options: QBBuildOptions, source: str) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "selected_path": source,
        "compile_options": options.compile_options_metadata(),
    }


def _legacy_fallback_metadata(cfg: BuildConfig) -> Dict[str, Any]:
    used = cfg.backend_options is None and bool(cfg.extra)
    return {
        "legacy_extra_fallback_used": used,
        "legacy_extra_keys": sorted(dict(cfg.extra or {}).keys()) if used else [],
    }


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified(cfg)",
        "backend": "qb",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


class _QBBuildAdapter:
    """Mobilint ARISE(QB) build adapter.

    두 가지 경로를 지원한다 (fetch 기본 + compile 훅):
      1) 이미 컴파일된 .mxq 를 제공받은 경우 -> 검증 후 out_dir 로 배치 (provided/fetch)
      2) ONNX / torch 모델을 받은 경우      -> compiler Python API(mxq_compile) 로 .mxq 컴파일
    """

    name = "qb"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"QB build adapter received backend={cfg.backend!r}")

        options = resolve_qb_build_options(cfg.backend_options, cfg.extra)
        mxq_path = _build_output_path(cfg.out_dir, cfg.model_name)
        mxq_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_input = prepare_qb_build_input(cfg.model_or_path, mxq_path)

        # ---- Path 1: 사전 컴파일된 .mxq 제공 (fetch / provided) ----
        if prepared_input.kind == "provided_artifact":
            artifact = prepared_input.provided_artifact
            if artifact is None:
                raise RuntimeError("QB prepare step returned an empty provided_artifact payload")
            placed_path = place_provided_qb_artifact(artifact)
            meta: Dict[str, Any] = {
                "backend": self.name,
                "mxq_path": str(placed_path),
                "source": "provided",
                "origin": str(artifact.source_path),
                "backend_options": options.compile_options_metadata(),
                **_legacy_fallback_metadata(cfg),
                **_capability_metadata(options, "provided"),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(placed_path),
                meta_data=meta,
            )

        # ---- Path 2: compiler Python API 로 ONNX/torch -> .mxq 컴파일 ----
        compile_source = prepared_input.compile_source
        if compile_source is None:
            raise RuntimeError("QB prepare step returned an empty compile_source payload")
        try:
            compiler_module_name, mxq_compile = _resolve_mxq_compile()
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(str(exc)) from exc

        _validate_shape(tuple(cfg.input_shape), "input_shape")
        quantize_method = options.quantize_method
        use_random_calib = options.use_random_calib
        if use_random_calib is None:
            use_random_calib = options.calib_data_path is None

        if options.calib_data_path is None and not use_random_calib:
            raise ValueError(
                "QB compile requires either QBBuildOptions.calib_data_path or "
                "QBBuildOptions.use_random_calib=True"
            )

        compile_kwargs: Dict[str, Any] = {
            "model": compile_source.source,      # ONNX 경로(str) 또는 torch 모델 인스턴스
            "save_path": str(mxq_path),
            "quantize_method": quantize_method,
            "use_random_calib": use_random_calib,
            "target_device": options.resolved_target_device(),
        }
        if options.calib_data_path:
            compile_kwargs["calib_data_path"] = str(options.calib_data_path)

        # 선택 옵션은 있으면 그대로 compiler Python API 로 패스스루
        for opt in ("model_nickname", "optimize_option", "singlecore_compile", "save_sample"):
            value = getattr(options, opt)
            if value is not None:
                compile_kwargs[opt] = value

        try:
            mxq_compile(**compile_kwargs)
        except Exception as exc:
            raise RuntimeError(f"{compiler_module_name} mxq_compile failed: {exc}") from exc

        if not mxq_path.is_file():
            raise RuntimeError(
                f"{compiler_module_name} reported success but .mxq not found at {mxq_path}. "
                f"Check {compiler_module_name} 'save_path' behavior for your compiler version."
            )

        meta = {
            "backend": self.name,
            "mxq_path": str(mxq_path),
            "source": f"{compiler_module_name}_compile",
            "prepared_source": compile_source.source_label,
            "compiler_module": compiler_module_name,
            "quantize_method": quantize_method,
            "use_random_calib": use_random_calib,
            "calib_data_path": options.calib_data_path,
            "input_shape": tuple(cfg.input_shape),
            "precision": "int8",
            "backend_options": options.compile_options_metadata(),
            **_legacy_fallback_metadata(cfg),
            **_capability_metadata(options, f"{compiler_module_name}_compile"),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(mxq_path),
            meta_data=meta,
        )


register(_QBBuildAdapter())
