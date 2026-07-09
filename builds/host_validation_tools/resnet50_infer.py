from __future__ import annotations

import argparse
import timeit
import traceback
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run host-native inference with a compiled TensorRT ResNet50 engine.")
    parser.add_argument(
        "--engine-path",
        type=Path,
        default=Path("builds/host_validation_outputs/host_resnet50.engine"),
    )
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    return parser


def _parse_shape(value: str) -> tuple:
    shape = tuple(int(p.strip()) for p in value.replace("x", ",").split(",") if p.strip())
    if not shape or any(d <= 0 for d in shape):
        raise ValueError(f"invalid input shape: {value!r}")
    return shape


def main() -> int:
    args = _build_parser().parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    import numpy as np

    engine_path = args.engine_path.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"engine not found: {engine_path}")

    print("== TensorRT ResNet50 Inference ==")
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except Exception:
        traceback.print_exc()
        print("Run inside the NVIDIA TensorRT container with pycuda installed.")
        return 1

    print("tensorrt =", getattr(trt, "__version__", "unknown"))
    shape = _parse_shape(args.input_shape)

    d_input = d_output = None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        context = engine.create_execution_context()

        if hasattr(context, "set_input_shape"):
            context.set_input_shape(args.input_name, shape)
        else:
            context.set_binding_shape(engine.get_binding_index(args.input_name), shape)

        if hasattr(context, "get_tensor_shape"):
            out_shape = tuple(context.get_tensor_shape(args.output_name))
        else:
            out_shape = tuple(context.get_binding_shape(engine.get_binding_index(args.output_name)))
        print("engine =", engine_path)
        print("input =", args.input_name, shape)
        print("output =", args.output_name, out_shape)

        h_input = cuda.pagelocked_empty(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.float32).reshape(out_shape)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()

        use_v3 = hasattr(context, "execute_async_v3")
        if use_v3:
            context.set_tensor_address(args.input_name, int(d_input))
            context.set_tensor_address(args.output_name, int(d_output))
            bindings = None
        else:
            bindings = [0] * engine.num_bindings
            bindings[engine.get_binding_index(args.input_name)] = int(d_input)
            bindings[engine.get_binding_index(args.output_name)] = int(d_output)

        h_input[...] = np.zeros(shape, dtype=np.float32)

        def _once():
            cuda.memcpy_htod_async(d_input, h_input, stream)
            if use_v3:
                context.execute_async_v3(stream_handle=stream.handle)
            else:
                context.execute_v2(bindings)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()

        for _ in range(args.warmup):
            _once()

        times_ms = []
        for _ in range(args.iters):
            t0 = timeit.default_timer()
            _once()
            times_ms.append((timeit.default_timer() - t0) * 1000.0)

        out = np.array(h_output, copy=True)
        flat = out.reshape(out.shape[0], -1) if out.ndim >= 2 else out.reshape(1, -1)
        cls_id = int(np.argmax(flat[0]))
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        for buf in (d_input, d_output):
            free = getattr(buf, "free", None)
            if callable(free):
                try:
                    free()
                except Exception:
                    pass

    print("input_source = synthetic zeros", shape)
    print("output_shape =", out.shape)
    print("pred_id =", cls_id)
    print(f"execute_v3 = {use_v3}")
    print(f"latency_ms_avg = {np.mean(times_ms):.3f}")
    print(f"latency_ms_min = {np.min(times_ms):.3f}")
    print(f"latency_ms_max = {np.max(times_ms):.3f}")
    print(f"OK: inference completed for {args.iters} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
