# TensorRT 호스트 검증

이 폴더는 의도적으로 `builds/` 아래(=gitignore)에 두어 저장소에 커밋되지 않습니다.
벤더 에스컬레이션용 **로컬/서버 전용 재현 팩**으로 사용하세요.

흐름은 `rbln-only` 호스트 검증 팩과 동일하며(env → smoke → resnet50 compile → infer),
NVIDIA TensorRT 툴체인(`tensorrt` + `pycuda`)에 맞춰 구성했습니다.
컴파일은 **ONNX → `.engine`**, 추론은 **deserialize + PyCUDA** 경로입니다.

## 호스트 Python 준비

가장 간단한 방법은 NVIDIA TensorRT 컨테이너를 쓰는 것입니다.

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/unified-sdk -w /workspace/unified-sdk \
  nvcr.io/nvidia/tensorrt:24.03-py3 bash
```

컨테이너 안에서:

```bash
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install onnx pycuda
# tensorrt 는 베이스 이미지에 포함되어 있습니다.
```

GPU/드라이버 확인:

```bash
nvidia-smi
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## 전체 검증 실행

```bash
python builds/host_validation_tools/run_host_validation.py
```

러너는 타임스탬프별 로그를 아래에 기록합니다:

```text
builds/host_validation_tools/logs/
```

컴파일 산출물은 아래에 저장됩니다:

```text
builds/host_validation_outputs/
```

## 개별 단계 실행

```bash
python builds/host_validation_tools/collect_env.py
python builds/host_validation_tools/smoke_conv_compile.py
python builds/host_validation_tools/resnet50_compile.py
python builds/host_validation_tools/resnet50_infer.py
```

`resnet50_infer.py` 는 `resnet50_compile.py` 가 만든 엔진을 기본값으로 사용합니다:

```text
builds/host_validation_outputs/host_resnet50.engine
```

## 참고

- `tensorrt` 는 ONNX 를 입력으로 받으므로, compile 단계는 먼저 torch → ONNX 로 내보낸 뒤
  `Builder` + `OnnxParser` + `build_serialized_network` 로 `.engine` 을 만듭니다.
- TRT 8.5+/10 은 `execute_async_v3` + `set_tensor_address`, 구버전은 `execute_v2` + bindings 를 사용합니다.
  스크립트는 양쪽을 자동 감지합니다.
- device 버퍼(`cuda.mem_alloc`)는 GC 에 맡기지 않고 `free()` 로 명시적으로 반환합니다.
- INT8 은 calibrator 가 필요합니다(본 검증 팩은 fp32/fp16 만 다룹니다).
