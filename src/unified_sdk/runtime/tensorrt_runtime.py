from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unified_sdk.options import resolve_tensorrt_vision_runtime_options
from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle


_CAPABILITY_FAMILY = "vision.low-level-engine-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "deserialize_engine",
    "create_execution_context",
    "allocate_host_and_device_buffers",
    "bind_tensor_addresses",
    "copy_h2d_execute_copy_d2h",
    "free_device_buffers",
)
_VENDOR_API_MAP = {
    "deserialize_engine": "trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)",
    "create_context": "engine.create_execution_context()",
    "allocate_buffers": "pycuda.driver.pagelocked_empty / mem_alloc / Stream",
    "bind_tensors": "context.set_tensor_address(...) or v2 bindings",
    "infer": "context.execute_async_v3(stream_handle=...) or context.execute_v2(bindings)",
    "destroy": "device_buffer.free() best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "trt.Runtime(...).deserialize_cuda_engine(engine_bytes)": "create_runtime(cfg)",
    "engine.create_execution_context()": "create_runtime(cfg)",
    "pycuda.pagelocked_empty / mem_alloc / Stream": "create_runtime(cfg)",
    "context.set_tensor_address(...) or v2 bindings": "create_runtime(cfg)",
    "context.execute_async_v3(...) / execute_v2(...)": "infer(rh, input_array)",
    "cuda.memcpy_htod_async / memcpy_dtoh_async": "infer(rh, input_array)",
    "device_buffer.free()": "destroy_runtime(rh)",
}


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime(cfg)",
            "infer": "infer(rh, input_array)",
            "destroy": "destroy_runtime(rh)",
        },
        "backend": "tensorrt",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RuntimeConfig.{field_name} must be a non-empty string")
    return value.strip()


def _validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"RuntimeConfig.{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"RuntimeConfig.{field_name} must contain only positive integers: {shape!r}")
    return shape


def _dtype_from_trt(trt, dtype) -> Any:
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
    }
    for key in ("BOOL", "INT64", "UINT8"):
        member = getattr(trt.DataType, key, None)
        if member is not None:
            mapping[member] = {"BOOL": np.bool_, "INT64": np.int64, "UINT8": np.uint8}[key]
    return mapping.get(dtype, np.float32)


def _tensor_dtype(trt, engine, name: str):
    if hasattr(engine, "get_tensor_dtype"):
        return _dtype_from_trt(trt, engine.get_tensor_dtype(name))
    idx = engine.get_binding_index(name)
    return _dtype_from_trt(trt, engine.get_binding_dtype(idx))


def _tensor_shape(engine, context, name: str) -> Tuple[int, ...]:
    if hasattr(context, "get_tensor_shape"):
        return tuple(context.get_tensor_shape(name))
    idx = engine.get_binding_index(name)
    return tuple(context.get_binding_shape(idx))


def _load_engine(trt, engine_path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    return engine, runtime


def _nbytes(shape: Tuple[int, ...], dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


class _TensorRTRuntime:
    name = "tensorrt"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT runtime adapter received backend={cfg.backend!r}")

        engine_path = Path(cfg.engine_path).expanduser().resolve()
        if not engine_path.is_file():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        if engine_path.suffix.lower() != ".engine":
            raise ValueError(f"Expected a .engine file, got: {engine_path}")

        input_name = _require_non_empty_string(cfg.input_name, "input_name")
        output_name = _require_non_empty_string(cfg.output_name, "output_name")
        input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")
        options = resolve_tensorrt_vision_runtime_options(cfg.backend_options)

        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except Exception as exc:
            raise RuntimeError(
                "tensorrt and pycuda are required for TensorRT inference. "
                "Use the NVIDIA TensorRT container or install them."
            ) from exc

        engine, trt_runtime = _load_engine(trt, engine_path)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        if hasattr(context, "set_input_shape"):
            context.set_input_shape(input_name, input_shape)
        else:
            context.set_binding_shape(engine.get_binding_index(input_name), input_shape)

        in_dtype = _tensor_dtype(trt, engine, input_name)
        out_dtype = _tensor_dtype(trt, engine, output_name)
        out_shape = _tensor_shape(engine, context, output_name)
        if any(d <= 0 for d in out_shape):
            raise RuntimeError(f"Output shape not resolved for {output_name!r}: {out_shape}")

        h_input = cuda.pagelocked_empty(int(np.prod(input_shape)), dtype=in_dtype).reshape(input_shape)
        h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=out_dtype).reshape(out_shape)
        d_input = cuda.mem_alloc(_nbytes(input_shape, in_dtype))
        d_output = cuda.mem_alloc(_nbytes(out_shape, out_dtype))
        stream = cuda.Stream()

        use_v3 = bool(options.use_execute_v3) and hasattr(context, "execute_async_v3")
        bindings: Optional[List[int]] = None
        if use_v3:
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
        else:
            bindings = [0] * engine.num_bindings
            bindings[engine.get_binding_index(input_name)] = int(d_input)
            bindings[engine.get_binding_index(output_name)] = int(d_output)

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(engine_path),
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            ctx={
                "cuda": cuda,
                "trt_runtime": trt_runtime,
                "engine": engine,
                "context": context,
                "stream": stream,
                "d_input": d_input,
                "d_output": d_output,
                "h_input": h_input,
                "h_output": h_output,
                "bindings": bindings,
                "use_v3": use_v3,
                "allow_dynamic_shape": bool(options.allow_dynamic_shape),
                "runtime_options": options.to_metadata(),
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        if not rh.ctx or "context" not in rh.ctx:
            raise RuntimeError("TensorRT RuntimeHandle is closed or invalid")

        ctx = rh.ctx
        cuda = ctx["cuda"]
        allow_dynamic = bool(ctx.get("allow_dynamic_shape", False))
        if (not allow_dynamic) and tuple(input_array.shape) != tuple(rh.input_shape):
            raise ValueError(f"Bad input shape: {input_array.shape}, expected {rh.input_shape}")
        if allow_dynamic and tuple(input_array.shape) != tuple(rh.input_shape):
            raise NotImplementedError(
                "allow_dynamic_shape=True currently bypasses only the static shape gate. "
                "End-to-end runtime rebind/reallocation for changed input shapes is not implemented in trt-only yet."
            )

        ctx["h_input"][...] = input_array.astype(ctx["h_input"].dtype, copy=False)
        try:
            cuda.memcpy_htod_async(ctx["d_input"], ctx["h_input"], ctx["stream"])
            if ctx["use_v3"]:
                ctx["context"].execute_async_v3(stream_handle=ctx["stream"].handle)
            else:
                ctx["context"].execute_v2(ctx["bindings"])
            cuda.memcpy_dtoh_async(ctx["h_output"], ctx["d_output"], ctx["stream"])
            ctx["stream"].synchronize()
        except Exception as exc:
            raise RuntimeError(f"TensorRT inference failed: {exc}") from exc
        return np.array(ctx["h_output"], copy=True)

    def destroy(self, rh: RuntimeHandle) -> None:
        ctx = rh.ctx or {}
        for key in ("d_input", "d_output"):
            buf = ctx.get(key)
            free = getattr(buf, "free", None)
            if callable(free):
                try:
                    free()
                except Exception:
                    pass
        ctx.clear()


register(_TensorRTRuntime())
