# examples/inspect_engine_io.py
from pathlib import Path

# TensorRT가 설치되지 않은 환경에서도 import 에러가 나지 않도록 처리
try:
    import tensorrt as trt
except ImportError:
    print("Error: 'tensorrt' module not found. Please install NVIDIA TensorRT.")
    sys.exit(1)


def inspect(engine_path: str):
    
    if not Path(engine_path).exists():
        print(f"Error: File not found - {engine_path}")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None

    print(f"\n== Engine: {Path(engine_path).name} ==")
    #print(f"  explicitBatch: {engine.has_implicit_batch_dimension() is False}")

    # TRT 8/10 모두 커버: v3 API 우선, 미지원 시 v2 바인딩 API 사용
    has_v3 = hasattr(engine, "get_tensor_name") and hasattr(engine, "get_tensor_mode")
    if has_v3:
        # v3 API
        n = engine.num_io_tensors
        print(f"  num_io_tensors: {n}")
        for i in range(n):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)  # trt.TensorIOMode.INPUT/OUTPUT
            dtype = engine.get_tensor_dtype(name)
            print(f"  - {i}: name={name!r}, mode={mode}, dtype={dtype}")
    else:
        # v2 API
        nb = engine.num_bindings
        print(f"  num_bindings: {nb}")
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            print(f"  - {i}: name={name!r}, {'INPUT' if is_input else 'OUTPUT'}, dtype={dtype}")


if __name__ == "__main__":
    # YOLOv7로 빌드한 엔진 경로로 바꿔서 실행하세요.
    inspect("build_output/yolov7_FP32.engine")
