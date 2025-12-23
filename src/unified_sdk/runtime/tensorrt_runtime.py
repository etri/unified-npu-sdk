from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda  # type: ignore
import pycuda.autoinit  # noqa: F401

from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle

def _dtype_from_trt(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT: return np.float32
    if dtype == trt.DataType.HALF:  return np.float16
    if dtype == trt.DataType.INT32: return np.int32
    if dtype == trt.DataType.INT8:  return np.int8
    return np.float32

def _load_engine(engine_path: Path) -> trt.ICudaEngine:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        serialized = f.read()
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    return engine

class _TensorRTRuntime:
    name = "tensorrt"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        engine_path = Path(cfg.engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        engine = _load_engine(engine_path)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create execution context")

        # EXPLICIT_BATCH 입력 shape 적용
        context.set_input_shape(cfg.input_name, cfg.input_shape)

        # dtype/shape
        has_v3 = hasattr(context, "execute_async_v3") and cfg.use_execute_v3
        in_dtype  = _dtype_from_trt(engine.get_tensor_dtype(cfg.input_name)  if hasattr(engine, "get_tensor_dtype")  else engine.get_binding_dtype(engine.get_binding_index(cfg.input_name)))
        out_dtype = _dtype_from_trt(engine.get_tensor_dtype(cfg.output_name) if hasattr(engine, "get_tensor_dtype") else engine.get_binding_dtype(engine.get_binding_index(cfg.output_name)))

        out_shape: Tuple[int, ...]
        if hasattr(engine, "get_tensor_shape"):
            out_shape = tuple(context.get_tensor_shape(cfg.output_name))  # type: ignore
        else:
            out_shape = tuple(context.get_binding_shape(engine.get_binding_index(cfg.output_name)))  # type: ignore

        # host pinned
        h_input  = cuda.pagelocked_empty(int(np.prod(cfg.input_shape)), dtype=in_dtype).reshape(cfg.input_shape)  # type: ignore
        h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=out_dtype).reshape(out_shape)            # type: ignore

        # device
        def _bytes(shape, dtype): return int(np.prod(shape)) * np.dtype(dtype).itemsize
        d_input  = cuda.mem_alloc(_bytes(cfg.input_shape, in_dtype))
        d_output = cuda.mem_alloc(_bytes(out_shape,     out_dtype))

        stream = cuda.Stream()
        bindings: List[int] | None = None
        if has_v3:
            context.set_tensor_address(cfg.input_name,  int(d_input))
            context.set_tensor_address(cfg.output_name, int(d_output))
        else:
            bindings = [0] * engine.num_bindings
            bi = engine.get_binding_index(cfg.input_name)
            bo = engine.get_binding_index(cfg.output_name)
            bindings[bi] = int(d_input)
            bindings[bo] = int(d_output)

        ctx: Dict[str, Any] = {
            "engine": engine,
            "context": context,
            "stream": stream,
            "d_input": d_input,
            "d_output": d_output,
            "h_input": h_input,
            "h_output": h_output,
            "bindings": bindings,
            "use_v3": has_v3,
        }

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(engine_path),
            input_name=cfg.input_name,
            output_name=cfg.output_name,
            input_shape=cfg.input_shape,
            ctx=ctx,
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        ctx = rh.ctx
        # 입력 검증/복사
        if tuple(input_array.shape) != tuple(rh.input_shape):
            raise ValueError(f"Bad input shape: {input_array.shape}, expected {rh.input_shape}")
        ctx["h_input"][...] = input_array.astype(ctx["h_input"].dtype, copy=False)

        # H2D
        cuda.memcpy_htod_async(ctx["d_input"], ctx["h_input"], ctx["stream"])

        # Exec
        if ctx["use_v3"]:
            ctx["context"].execute_async_v3(stream_handle=ctx["stream"].handle)
        else:
            ctx["context"].execute_v2(ctx["bindings"])

        # D2H
        cuda.memcpy_dtoh_async(ctx["h_output"], ctx["d_output"], ctx["stream"])
        ctx["stream"].synchronize()

        return np.array(ctx["h_output"], copy=True)

    def destroy(self, rh: RuntimeHandle) -> None:
        ctx = rh.ctx
        for k in ["d_input", "d_output", "h_input", "h_output", "context", "engine", "stream"]:
            try:
                del ctx[k]
            except Exception:
                pass

# 자동 등록
register(_TensorRTRuntime())

