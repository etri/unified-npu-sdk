from __future__ import annotations

import argparse
import traceback
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a minimal Conv2d smoke model to a TensorRT engine.")
    parser.add_argument("--onnx", type=Path, default=Path("builds/host_validation_outputs/host_smoke_conv.onnx"))
    parser.add_argument("--output", type=Path, default=Path("builds/host_validation_outputs/host_smoke_conv.engine"))
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--workspace-mib", type=int, default=256)
    return parser


def _build_engine(trt, onnx_path: Path, engine_path: Path, *, input_name: str,
                  shape: tuple, precision: str, workspace_mib: int) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    explicit = getattr(flag_enum, "EXPLICIT_BATCH", None) if flag_enum else None
    network = builder.create_network(1 << int(explicit)) if explicit is not None else builder.create_network(0)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX:\n{errors}")

    config = builder.create_builder_config()
    pool = getattr(trt, "MemoryPoolType", None)
    if pool is not None and hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(pool.WORKSPACE, workspace_mib * (1 << 20))
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mib * (1 << 20)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


def main() -> int:
    args = _build_parser().parse_args()

    import torch
    import torch.nn as nn

    class Smoke(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 10),
            )

        def forward(self, x):
            return self.net(x)

    print("== TensorRT Smoke Conv Compile ==")
    print("torch =", torch.__version__)

    shape = (1, 3, 224, 224)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            Smoke().eval(), torch.zeros(*shape, dtype=torch.float32), str(args.onnx),
            input_names=["input"], output_names=["output"], opset_version=13, dynamo=False,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print("onnx =", args.onnx)

    try:
        import tensorrt as trt
    except Exception:
        traceback.print_exc()
        print("Run inside the NVIDIA TensorRT container (nvcr.io/nvidia/tensorrt).")
        return 1
    print("tensorrt =", getattr(trt, "__version__", "unknown"))

    try:
        _build_engine(trt, args.onnx, args.output, input_name="input", shape=shape,
                      precision=args.precision, workspace_mib=args.workspace_mib)
    except Exception:
        traceback.print_exc()
        return 1

    if not args.output.is_file():
        print(f"ERROR: engine not found at {args.output}")
        return 1

    print(f"OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
