from __future__ import annotations

import shutil
import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

from unified_sdk.build.registry import register
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
    "provided_artifact": "shutil.copyfile(src_mxq, mxq_path)",
    "compile": "compiler_python_api.mxq_compile(**compile_kwargs)",
    "calibration": "calib_data_path or use_random_calib",
    "artifact": ".mxq",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "shutil.copyfile(src_mxq, mxq_path)": "build_unified(cfg) for provided .mxq",
    "compiler_python_api.mxq_compile(**compile_kwargs)": "build_unified(cfg) for ONNX/torch compile",
    "calib_data_path or use_random_calib": "BuildConfig.calib_data_path / BuildConfig.extra",
    ".mxq artifact": "BuildResult.compiled_model_path",
}


# Mobilint compiler Python API(mxq_compile) 가 지원하는 양자화 방법
_QUANTIZE_METHODS = ("percentile", "maxpercentile", "max", "kl")
_TARGET_DEVICE_BY_PRODUCT = {
    "aries": "aries-rb",
    "aries-rb": "aries-rb",
    "regulus": "regulus-rb",
    "regulus-rb": "regulus-rb",
    "regulus-ra": "regulus-ra",
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


def _validate_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    quantize_method = extra.get("quantize_method")
    if quantize_method is not None and quantize_method not in _QUANTIZE_METHODS:
        raise ValueError(
            "BuildConfig.extra['quantize_method'] must be one of: "
            + ", ".join(repr(m) for m in _QUANTIZE_METHODS)
        )
    core_mode = extra.get("core_mode")
    if core_mode is not None and (not isinstance(core_mode, str) or not core_mode.strip()):
        raise ValueError("BuildConfig.extra['core_mode'] must be a non-empty string when provided")
    target_device = extra.get("target_device")
    if target_device is not None and (not isinstance(target_device, str) or not target_device.strip()):
        raise ValueError("BuildConfig.extra['target_device'] must be a non-empty string when provided")
    return extra


def _build_output_path(out_dir: str | Path, model_name: str) -> Path:
    name = _require_non_empty_string(model_name, "model_name")
    if name.lower().endswith(".mxq"):
        return Path(out_dir) / name
    return Path(out_dir) / f"{name}.mxq"


def _resolve_target_device(extra: Dict[str, Any]) -> str:
    explicit = extra.get("target_device")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    product = str(extra.get("product", "aries")).strip().lower()
    return _TARGET_DEVICE_BY_PRODUCT.get(product, product)


def _looks_like_mxq(model_or_path: Any) -> bool:
    return isinstance(model_or_path, (str, Path)) and str(model_or_path).endswith(".mxq")


def _capability_metadata(extra: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "selected_path": source,
        "compile_options": {
            "quantize_method": extra.get("quantize_method", "percentile"),
            "use_random_calib": extra.get("use_random_calib"),
            "model_nickname": extra.get("model_nickname"),
            "optimize_option": extra.get("optimize_option"),
            "singlecore_compile": extra.get("singlecore_compile"),
            "save_sample": extra.get("save_sample"),
        },
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

        if cfg.precision not in ("int8",):
            raise ValueError(f"Unsupported QB precision: {cfg.precision!r} (MXQ is int8-quantized)")

        extra = _validate_extra(dict(cfg.extra or {}))
        mxq_path = _build_output_path(cfg.out_dir, cfg.model_name)
        mxq_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- Path 1: 사전 컴파일된 .mxq 제공 (fetch / provided) ----
        if _looks_like_mxq(cfg.model_or_path):
            src = Path(cfg.model_or_path).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Provided .mxq not found: {src}")
            if src != mxq_path.resolve():
                shutil.copyfile(src, mxq_path)
            meta: Dict[str, Any] = {
                "backend": self.name,
                "mxq_path": str(mxq_path),
                "source": "provided",
                "origin": str(src),
                "precision": cfg.precision,
                "extra": extra,
                **_capability_metadata(extra, "provided"),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(mxq_path),
                meta_data=meta,
            )

        # ---- Path 2: compiler Python API 로 ONNX/torch -> .mxq 컴파일 ----
        try:
            compiler_module_name, mxq_compile = _resolve_mxq_compile()
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(str(exc)) from exc

        _validate_shape(tuple(cfg.input_shape), "input_shape")
        quantize_method = extra.get("quantize_method", "percentile")
        use_random_calib = bool(extra.get("use_random_calib", cfg.calib_data_path is None))

        if cfg.calib_data_path is None and not use_random_calib:
            raise ValueError(
                "QB compile requires either BuildConfig.calib_data_path or "
                "BuildConfig.extra['use_random_calib']=True"
            )

        compile_kwargs: Dict[str, Any] = {
            "model": cfg.model_or_path,          # ONNX 경로(str) 또는 torch 모델 인스턴스
            "save_path": str(mxq_path),
            "quantize_method": quantize_method,
            "use_random_calib": use_random_calib,
            "target_device": _resolve_target_device(extra),
        }
        if cfg.calib_data_path:
            compile_kwargs["calib_data_path"] = str(cfg.calib_data_path)

        # 선택 옵션은 있으면 그대로 compiler Python API 로 패스스루
        for opt in ("model_nickname", "optimize_option", "singlecore_compile", "save_sample"):
            if extra.get(opt) is not None:
                compile_kwargs[opt] = extra[opt]

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
            "compiler_module": compiler_module_name,
            "quantize_method": quantize_method,
            "use_random_calib": use_random_calib,
            "calib_data_path": cfg.calib_data_path,
            "input_shape": tuple(cfg.input_shape),
            "precision": cfg.precision,
            "extra": extra,
            **_capability_metadata(extra, f"{compiler_module_name}_compile"),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(mxq_path),
            meta_data=meta,
        )


register(_QBBuildAdapter())
