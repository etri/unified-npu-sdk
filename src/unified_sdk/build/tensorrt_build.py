from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from unified_sdk.build.registry import register
from unified_sdk.frontends.types import PreparedTensorRTCompileSource, PreparedTensorRTVisionBuildInput
from unified_sdk.options import resolve_tensorrt_vision_build_options
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "vision.low-level-engine-builder"
_BUILD_PIPELINE = (
    "resolve_prepared_input",
    "validate_build_options",
    "copy_provided_artifact_or_parse_onnx_network",
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
    "builder.create_optimization_profile(); profile.set_shape(...)": "PreparedTensorRTCompileSource profile",
    "config.set_flag(trt.BuilderFlag.FP16/INT8)": "TensorRTVisionBuildOptions.precision",
    "builder.build_serialized_network(network, config)": "build_unified(cfg)",
    ".engine artifact": "BuildResult.compiled_model_path",
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


def _validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"{field_name} must contain only positive integers: {shape!r}")
    return shape


def _validate_profile(source: PreparedTensorRTCompileSource) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    lo = _validate_shape(tuple(source.min_input_shape), "PreparedTensorRTCompileSource.min_input_shape")
    opt = _validate_shape(tuple(source.opt_input_shape), "PreparedTensorRTCompileSource.opt_input_shape")
    hi = _validate_shape(tuple(source.max_input_shape), "PreparedTensorRTCompileSource.max_input_shape")
    if not (len(lo) == len(opt) == len(hi)):
        raise ValueError(f"min/opt/max_input_shape rank mismatch: {lo} / {opt} / {hi}")
    for i, (a, b, c) in enumerate(zip(lo, opt, hi)):
        if not (a <= b <= c):
            raise ValueError(f"optimization profile must satisfy min<=opt<=max at dim {i}: {a}/{b}/{c}")
    return lo, opt, hi


def _set_workspace(config, trt, workspace_mib: int | None) -> None:
    if not workspace_mib:
        return
    nbytes = workspace_mib * (1 << 20)
    pool_type = getattr(trt, "MemoryPoolType", None)
    if pool_type is not None and hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(pool_type.WORKSPACE, nbytes)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = nbytes


def _create_network(builder, trt):
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    explicit = getattr(flag_enum, "EXPLICIT_BATCH", None) if flag_enum else None
    if explicit is not None:
        try:
            return builder.create_network(1 << int(explicit))
        except Exception:
            pass
    return builder.create_network(0)


def _compile_tensorrt_engine(
    trt,
    *,
    source: PreparedTensorRTCompileSource,
    engine_path: Path,
    precision: str,
    workspace_mib: int | None,
    int8_calibrator: Any,
) -> Path:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = _create_network(builder, trt)
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(str(source.source_path))
    if not ok:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX: {source.source_path}\n{errors}")

    config = builder.create_builder_config()
    _set_workspace(config, trt, workspace_mib)

    lo, opt, hi = _validate_profile(source)
    profile = builder.create_optimization_profile()
    profile.set_shape(source.input_name, lo, opt, hi)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("precision='fp16' requested but this platform has no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("precision='int8' requested but this platform has no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = int8_calibrator
        if hasattr(config, "set_calibration_profile"):
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
    name = "tensorrt"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT build adapter received backend={cfg.backend!r}")

        options = resolve_tensorrt_vision_build_options(cfg.backend_options)
        precision = options.precision
        if precision == "int8" and options.int8_calibrator is None:
            raise ValueError(
                "precision='int8' requires a calibrator. "
                "Pass TensorRTVisionBuildOptions(int8_calibrator=...)."
            )

        prepared_input = cfg.prepared_input
        output_path = Path(cfg.out_dir).expanduser().resolve() / f"{cfg.model_name}_{precision.upper()}.engine"

        if prepared_input is None:
            model_path = Path(str(cfg.model_or_path)).expanduser().resolve()
            if model_path.suffix.lower() != ".engine" or not model_path.is_file():
                raise ValueError(
                    "TensorRT compile requires a prepared frontend contract. "
                    "Use resolve_tensorrt_vision_build_request(...) and pass BuildConfig.prepared_input."
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path != output_path:
                output_path.write_bytes(model_path.read_bytes())
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(output_path),
                meta_data={
                    "backend": self.name,
                    "model_name": cfg.model_name,
                    "precision": precision,
                    "source": "legacy_direct_provided_artifact",
                    "origin_engine_path": str(model_path),
                    "engine_path": str(output_path),
                    "backend_options": options.to_metadata(),
                    "capability_family": _CAPABILITY_FAMILY,
                    "build_pipeline": _BUILD_PIPELINE,
                    "vendor_api_map": _VENDOR_API_MAP,
                },
            )

        if prepared_input.kind == "provided_artifact":
            provided = prepared_input.provided_artifact
            if provided is None:
                raise ValueError("PreparedTensorRTVisionBuildInput.kind='provided_artifact' requires provided_artifact")
            src_engine = provided.source_path.expanduser().resolve()
            if not src_engine.is_file():
                raise FileNotFoundError(f"Engine file not found: {src_engine}")
            engine_path = provided.destination_path.expanduser().resolve()
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            if src_engine != engine_path:
                engine_path.write_bytes(src_engine.read_bytes())
            meta = {
                "backend": self.name,
                "model_name": cfg.model_name,
                "precision": precision,
                "source": "provided",
                "origin_engine_path": str(src_engine),
                "engine_path": str(engine_path),
                "prepared_kind": prepared_input.kind,
                "backend_options": options.to_metadata(),
                "capability_family": _CAPABILITY_FAMILY,
                "build_pipeline": _BUILD_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            }
            return BuildResult(backend=self.name, compiled_model_path=str(engine_path), meta_data=meta)

        compile_source = prepared_input.compile_source
        if compile_source is None:
            raise ValueError("PreparedTensorRTVisionBuildInput.kind='compile_source' requires compile_source")

        try:
            import tensorrt as trt
        except Exception as exc:
            raise RuntimeError(
                "tensorrt is required to build an engine. Use the NVIDIA TensorRT container or install the tensorrt package."
            ) from exc

        compiled = _compile_tensorrt_engine(
            trt,
            source=compile_source,
            engine_path=output_path,
            precision=precision,
            workspace_mib=options.workspace_mib,
            int8_calibrator=options.int8_calibrator,
        )
        meta = {
            "backend": self.name,
            "model_name": cfg.model_name,
            "precision": precision,
            "engine_path": str(compiled),
            "prepared_kind": prepared_input.kind,
            "source_label": compile_source.source_label,
            "provenance_kind": compile_source.provenance_kind,
            "provenance_detail": compile_source.provenance_detail,
            "source_path": str(compile_source.source_path),
            "profile": {
                "input_name": compile_source.input_name,
                "min": compile_source.min_input_shape,
                "opt": compile_source.opt_input_shape,
                "max": compile_source.max_input_shape,
            },
            "trt_version": getattr(trt, "__version__", "unknown"),
            "backend_options": options.to_metadata(),
            "capability_family": _CAPABILITY_FAMILY,
            "build_pipeline": _BUILD_PIPELINE,
            "vendor_api_map": _VENDOR_API_MAP,
        }
        return BuildResult(backend=self.name, compiled_model_path=str(compiled), meta_data=meta)


register(_TensorRTBuildAdapter())
