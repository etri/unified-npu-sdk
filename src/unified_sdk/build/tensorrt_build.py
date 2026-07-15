from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from unified_sdk.build.registry import register
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "vision.low-level-engine-builder"
_BUILD_PIPELINE = (
    "validate_config",
    "parse_onnx_network",
    "configure_builder",
    "configure_optimization_profile",
    "run_serialized_engine_build",
    "save_artifact",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "provided_artifact": "Path(src_engine).read_bytes() -> engine_path.write_bytes(...)",
    "parse_network": "trt.OnnxParser(network, logger).parse_from_file(str(onnx_path))",
    "builder_config": "builder.create_builder_config()",
    "optimization_profile": "builder.create_optimization_profile(); profile.set_shape(...)",
    "precision_flags": "config.set_flag(trt.BuilderFlag.FP16/INT8)",
    "compile": "builder.build_serialized_network(network, config)",
    "artifact": ".engine",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "Path(src_engine).read_bytes() -> engine_path.write_bytes(...)": "build_unified(cfg) for provided .engine",
    "trt.OnnxParser(...).parse_from_file(...)": "build_unified(cfg)",
    "builder.create_builder_config()": "build_unified(cfg)",
    "builder.create_optimization_profile(); profile.set_shape(...)": "BuildConfig.min/opt/max_input_shape",
    "config.set_flag(trt.BuilderFlag.FP16/INT8)": "BuildConfig.precision",
    "builder.build_serialized_network(network, config)": "build_unified(cfg)",
    ".engine artifact": "BuildResult.compiled_model_path",
}


_PRECISIONS = ("fp32", "fp16", "int8")


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


def _validate_profile(cfg: BuildConfig) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    lo = _validate_shape(tuple(cfg.min_input_shape), "min_input_shape")
    opt = _validate_shape(tuple(cfg.opt_input_shape), "opt_input_shape")
    hi = _validate_shape(tuple(cfg.max_input_shape), "max_input_shape")
    if not (len(lo) == len(opt) == len(hi)):
        raise ValueError(f"min/opt/max_input_shape rank mismatch: {lo} / {opt} / {hi}")
    for i, (a, b, c) in enumerate(zip(lo, opt, hi)):
        if not (a <= b <= c):
            raise ValueError(f"optimization profile must satisfy min<=opt<=max at dim {i}: {a}/{b}/{c}")
    return lo, opt, hi


def _validate_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    workspace_mib = extra.get("workspace_mib")
    if workspace_mib is not None:
        if not isinstance(workspace_mib, int) or workspace_mib <= 0:
            raise ValueError("BuildConfig.extra['workspace_mib'] must be a positive integer (MiB)")
    return extra


def _ensure_onnx_path(model_or_path: str | Path) -> Path:
    p = Path(model_or_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"ONNX file not found: {p}")
    if p.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX file, got: {p.suffix}")
    return p.resolve()


def _ensure_engine_path(model_or_path: str | Path) -> Path:
    p = Path(model_or_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Engine file not found: {p}")
    if p.suffix.lower() != ".engine":
        raise ValueError(f"Expected a .engine file, got: {p.suffix}")
    return p.resolve()


def _build_output_path(out_dir: str | Path, model_name: str, precision: str) -> Path:
    name = _require_non_empty_string(model_name, "model_name")
    return Path(out_dir) / f"{name}_{precision.upper()}.engine"


def _capability_metadata(extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "compile_options": {
            "workspace_mib": extra.get("workspace_mib"),
            "strict_types": extra.get("strict_types"),
            "int8_calibrator": "<provided>" if extra.get("int8_calibrator") is not None else None,
        },
    }


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified(cfg)",
        "backend": "tensorrt",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _create_network(builder, trt):
    """EXPLICIT_BATCH 플래그는 TRT 10 에서 제거/기본화되었다. 양쪽을 모두 지원한다."""
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    explicit = getattr(flag_enum, "EXPLICIT_BATCH", None) if flag_enum else None
    if explicit is not None:
        try:
            return builder.create_network(1 << int(explicit))
        except Exception:
            pass
    return builder.create_network(0)


def _set_workspace(config, trt, workspace_mib: int | None) -> None:
    if not workspace_mib:
        return
    nbytes = workspace_mib * (1 << 20)
    # TRT 8.4+ : set_memory_pool_limit / 구버전 : max_workspace_size
    pool_type = getattr(trt, "MemoryPoolType", None)
    if pool_type is not None and hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(pool_type.WORKSPACE, nbytes)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = nbytes


def _compile_tensorrt_engine(
    trt,
    onnx_path: Path,
    engine_path: Path,
    *,
    input_name: str,
    min_input_shape: Tuple[int, ...],
    opt_input_shape: Tuple[int, ...],
    max_input_shape: Tuple[int, ...],
    precision: str,
    workspace_mib: int | None,
    int8_calibrator: Any,
) -> Path:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = _create_network(builder, trt)

    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(str(onnx_path))
    if not ok:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}\n{errors}")

    config = builder.create_builder_config()
    _set_workspace(config, trt, workspace_mib)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_input_shape, opt_input_shape, max_input_shape)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("precision='fp16' requested but this platform has no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("precision='int8' requested but this platform has no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 은 calibration 없이는 정확도를 보장할 수 없다.
        config.int8_calibrator = int8_calibrator
        if hasattr(config, "add_optimization_profile"):
            try:
                config.set_calibration_profile(profile)
            except Exception:
                pass

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None (see TRT logger output)")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path


class _TensorRTBuildAdapter:
    """TensorRT build adapter: ONNX -> serialized `.engine`.

    - `tensorrt` 는 build() 내부에서 lazy import 한다 (GPU 없는 환경에서도 패키지 import 가능).
    - dynamic shape 는 min/opt/max optimization profile 로 지정한다.
    - int8 은 `extra["int8_calibrator"]` 가 있어야 하며, 없으면 조용히 fp32 로 떨어지지 않고 거부한다.
    """

    name = "tensorrt"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT build adapter received backend={cfg.backend!r}")

        precision = str(cfg.precision).lower()
        if precision not in _PRECISIONS:
            raise ValueError(f"Unsupported TensorRT precision: {cfg.precision!r} (expected {_PRECISIONS})")

        input_name = _require_non_empty_string(cfg.input_name, "input_name")
        lo, opt, hi = _validate_profile(cfg)
        extra = _validate_extra(dict(cfg.extra or {}))
        workspace_mib = extra.get("workspace_mib")
        int8_calibrator = extra.get("int8_calibrator")

        if precision == "int8" and int8_calibrator is None:
            raise ValueError(
                "precision='int8' requires a calibrator. "
                "Pass BuildConfig.extra['int8_calibrator'] (trt.IInt8EntropyCalibrator2 등)."
            )

        engine_path = _build_output_path(cfg.out_dir, cfg.model_name, precision)

        if isinstance(cfg.model_or_path, (str, Path)) and str(cfg.model_or_path).endswith(".engine"):
            src_engine = _ensure_engine_path(cfg.model_or_path)
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            if src_engine != engine_path.resolve():
                engine_path.write_bytes(src_engine.read_bytes())
            meta: Dict[str, Any] = {
                "backend": self.name,
                "model_name": cfg.model_name,
                "precision": precision,
                "source": "provided",
                "origin_engine_path": str(src_engine),
                "engine_path": str(engine_path),
                "extra": extra,
                **_capability_metadata(extra),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(engine_path),
                meta_data=meta,
            )

        onnx_path = _ensure_onnx_path(cfg.model_or_path)

        try:
            import tensorrt as trt
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(
                "tensorrt is required to build an engine. "
                "Use the NVIDIA TensorRT container or install the tensorrt python package."
            ) from exc

        try:
            compiled = _compile_tensorrt_engine(
                trt,
                onnx_path=onnx_path,
                engine_path=engine_path,
                input_name=input_name,
                min_input_shape=lo,
                opt_input_shape=opt,
                max_input_shape=hi,
                precision=precision,
                workspace_mib=workspace_mib,
                int8_calibrator=int8_calibrator,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"TensorRT engine build failed for {onnx_path}: {exc}") from exc

        meta: Dict[str, Any] = {
            "backend": self.name,
            "model_name": cfg.model_name,
            "precision": precision,
            "onnx_path": str(onnx_path),
            "engine_path": str(compiled),
            "trt_version": getattr(trt, "__version__", "unknown"),
            "workspace_mib": workspace_mib,
            "profile": {
                "input_name": input_name,
                "min": lo,
                "opt": opt,
                "max": hi,
            },
            "extra": {k: v for k, v in extra.items() if k != "int8_calibrator"},
            **_capability_metadata(extra),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(compiled),
            meta_data=meta,
        )


register(_TensorRTBuildAdapter())
