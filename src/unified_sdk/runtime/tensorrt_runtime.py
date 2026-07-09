from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

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


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"RuntimeConfig.extra['{field_name}'] must be a boolean-like value, got {value!r}")


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
    serialized = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    # runtime 객체가 GC 되면 engine 이 무효화될 수 있으므로 함께 반환해 붙잡아 둔다.
    return engine, runtime


def _nbytes(shape: Tuple[int, ...], dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


class _TensorRTRuntime:
    """TensorRT runtime adapter (TensorRT + PyCUDA).

    - `tensorrt` / `pycuda` 는 create() 내부에서 lazy import 한다.
      특히 `pycuda.autoinit` 은 import 시점에 CUDA 컨텍스트를 만들기 때문에,
      모듈 최상단이 아니라 런타임 생성 시점에만 초기화되도록 미뤘다.
    - TRT 8.5+/10 은 execute_async_v3 + set_tensor_address, 구버전은 execute_v2 + bindings.
    - destroy() 에서 device 버퍼를 명시적으로 free 한다 (GC 의존 X).
    """

    name = "tensorrt"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT runtime adapter received backend={cfg.backend!r}")

        engine_path = Path(cfg.engine_path).expanduser()
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        if engine_path.suffix != ".engine":
            raise ValueError(f"Expected a .engine file, got: {engine_path}")

        input_name = _require_non_empty_string(cfg.input_name, "input_name")
        output_name = _require_non_empty_string(cfg.output_name, "output_name")
        input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")
        extra = dict(cfg.extra or {})

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401  (CUDA 컨텍스트 초기화 — lazy)
        except Exception as exc:  # pragma: no cover - GPU/벤더 SDK 필요
            raise RuntimeError(
                "tensorrt and pycuda are required for TensorRT inference. "
                "Use the NVIDIA TensorRT container (nvcr.io/nvidia/tensorrt) or install them."
            ) from exc

        engine, trt_runtime = _load_engine(trt, engine_path)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        # dynamic shape: 실행 shape 확정
        if hasattr(context, "set_input_shape"):
            context.set_input_shape(input_name, input_shape)
        else:  # 구버전
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

        use_v3 = bool(cfg.use_execute_v3) and hasattr(context, "execute_async_v3")
        bindings: Optional[List[int]] = None
        if use_v3:
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
        else:
            bindings = [0] * engine.num_bindings
            bindings[engine.get_binding_index(input_name)] = int(d_input)
            bindings[engine.get_binding_index(output_name)] = int(d_output)

        ctx: Dict[str, Any] = {
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
            "out_shape": out_shape,
            "extra": extra,
            "capability_family": _CAPABILITY_FAMILY,
            "runtime_pipeline": _RUNTIME_PIPELINE,
            "vendor_api_map": _VENDOR_API_MAP,
        }

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(engine_path),
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            ctx=ctx,
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        if not rh.ctx or "context" not in rh.ctx:
            raise RuntimeError("TensorRT RuntimeHandle is closed or invalid")

        ctx = rh.ctx
        cuda = ctx["cuda"]
        extra = ctx.get("extra", {})
        allow_dynamic = _parse_bool(extra.get("allow_dynamic_shape", False), "allow_dynamic_shape")

        if (not allow_dynamic) and tuple(input_array.shape) != tuple(rh.input_shape):
            raise ValueError(f"Bad input shape: {input_array.shape}, expected {rh.input_shape}")

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
        # 1) device 메모리 명시적 해제 (PyCUDA mem_alloc 은 GC 의존이 아니라 free() 로 반환)
        for key in ("d_input", "d_output"):
            buf = ctx.get(key)
            free = getattr(buf, "free", None)
            if callable(free):
                try:
                    free()
                except Exception:
                    pass
        # 2) 나머지 참조 해제 (context -> engine -> runtime 순)
        ctx.clear()


register(_TensorRTRuntime())
