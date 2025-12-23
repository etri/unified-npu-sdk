from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import tensorrt as trt

from unified_sdk.build.registry import register
from unified_sdk.types import BuildConfig, BuildResult

def _ensure_onnx_path(model_or_path: str | Path) -> Path:
    p = Path(model_or_path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX file not found: {p}")
    if p.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX file, got: {p.suffix}")
    return p

def _compile_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    input_name: str,
    min_input_shape: Tuple[int, ...],
    opt_input_shape: Tuple[int, ...],
    max_input_shape: Tuple[int, ...],
    use_fp16: bool,
) -> Path:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, TRT_LOGGER)
    ok = parser.parse_from_file(str(onnx_path))
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    if not ok:
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_input_shape, opt_input_shape, max_input_shape)

    config = builder.create_builder_config()
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path

class _TensorRTBuildAdapter:
    name = "tensorrt"

    def build(self, cfg: BuildConfig) -> BuildResult:
        onnx_path = _ensure_onnx_path(cfg.model_or_path)
        use_fp16 = (cfg.precision.lower() == "fp16")
        suffix = "FP16" if use_fp16 else "FP32"
        out_dir = Path(cfg.out_dir)
        engine_path = out_dir / f"{cfg.model_name}_{suffix}.engine"

        compiled = _compile_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_name=cfg.input_name,
            min_input_shape=cfg.min_input_shape,
            opt_input_shape=cfg.opt_input_shape,
            max_input_shape=cfg.max_input_shape,
            use_fp16=use_fp16,
        )

        meta: Dict[str, Any] = {
            "backend": self.name,
            "model_name": cfg.model_name,
            "precision": cfg.precision,
            "onnx_path": str(onnx_path),
            "engine_path": str(compiled),
            "profile": {
                "input_name": cfg.input_name,
                "min": cfg.min_input_shape,
                "opt": cfg.opt_input_shape,
                "max": cfg.max_input_shape,
            },
            "extra": cfg.extra or {},
        }
        return BuildResult(backend=self.name, compiled_model_path=str(compiled), meta_data=meta)

# 자동 등록
register(_TensorRTBuildAdapter())

