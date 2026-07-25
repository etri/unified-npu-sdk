# Unified SDK — TRT-only (NVIDIA TensorRT)

이 체크아웃(`trt-only` 브랜치)은 **NVIDIA TensorRT 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 TensorRT 1종으로 좁힌 버전입니다.

`rbln-only`·`qb-only`·`furiosa-only`와 동일한 단일-백엔드 패턴을 따릅니다.
컴파일은 **ONNX → `.engine`**(`Builder` + `OnnxParser` + `build_serialized_network`),
추론은 **TensorRT + PyCUDA**(`execute_async_v3` / `execute_v2`)를 사용합니다.
LLM 경로는 **TensorRT-LLM**(`tensorrt_llm.LLM`, `SamplingParams`, `llm.generate`)를 사용합니다.

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 TensorRT 단일 백엔드 분기입니다.
TensorRT 분기는 국산 NPU 백엔드들의 **비교 기준(reference)** 역할을 합니다.

---

## 🏗️ 프로젝트 구조

```
<repo-root>/
├── README.md
├── LICENSE
├── pyproject.toml
├── pyrightconfig.json
├── requirements.txt
├── devcontainer.json
├── Dockerfile
├── build.sh
├── examples/
│   ├── run_tensorrt_build.py       # ONNX → .engine 컴파일
│   ├── run_tensorrt_infer.py       # .engine 추론 + latency 측정
│   └── inspect_engine_io.py        # .engine 입출력 텐서 메타 확인
│   ├── run_tensorrt_llm_build.py   # model ref/path -> TensorRT-LLM fetch/compile
│   ├── run_tensorrt_llm_infer.py   # TensorRT-LLM generate
│   └── inspect_tensorrt_llm_model.py # TensorRT-LLM artifact/model ref 점검
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (TensorRT 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified / build_unified_LLM
    │   ├── registry.py
    │   └── tensorrt_build.py       # TensorRT 빌드 어댑터
    │   └── tensorrt_llm_build.py   # TensorRT-LLM 빌드/패스스루 어댑터
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / destroy_runtime / *_LLM
        ├── registry.py
        └── tensorrt_runtime.py     # TensorRT 런타임 어댑터 (PyCUDA)
        └── tensorrt_llm_runtime.py # TensorRT-LLM 런타임 어댑터
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. `rbln-only`와 동일한 흐름(env → smoke → resnet50
> compile → infer)을 TensorRT 기준으로 구성했습니다.

---

## 💾 설치 방법

### 1. 호스트 사전 요구사항

- **NVIDIA GPU 드라이버**가 호스트에 정상 설치되어 있어야 합니다.
- **Docker Engine**, **docker buildx 플러그인**, **NVIDIA Container Toolkit**이 준비되어 있어야 합니다.
- `tensorrt`는 NVIDIA 공식 컨테이너(`nvcr.io/nvidia/tensorrt`)에 포함되어 있어 별도 설치가 필요 없습니다.
- `tensorrt_llm`는 용량이 큰 편이라 설치 시간이 길 수 있습니다. 다만 `trt-only` 이미지는 vision/LLM 공용으로 재활용하는 전제를 두고, Docker 빌드 시 기본 포함합니다.
- 2026년 7월 25일 기준 `trt-only`는 `nvcr.io/nvidia/tensorrt:24.03-py3` 베이스와 맞추기 위해 `tensorrt_llm==0.10.0`, `torch==2.2.2`, `torchvision==0.17.2` 축으로 pin 합니다. 이는 NVIDIA TensorRT-LLM 0.10.0 릴리스 노트의 `NGC 24.03`, `TensorRT 10.0.1`, `CUDA 12.4`, `PyTorch 2.2.2` 의존성 축을 따른 것입니다.
- 자세한 내용은 <https://developer.nvidia.com/tensorrt> 참조.

최소 확인 항목:

```bash
nvidia-smi
docker --version
docker buildx version
```

아래가 모두 통과해야 `./build.sh` 까지 무리 없이 진행됩니다.

### 2. Docker 사전 준비

- `trt-only` 검증은 **Docker 기준**으로 진행합니다. 호스트에 `pip install -e .` 같은 로컬 직접 설치는 권장하지 않습니다.
- Ubuntu에서는 **Docker 공식 apt 저장소** 기준 설치를 권장합니다. `docker.io`만 설치하면 `buildx`가 없을 수 있습니다.
- 설치 후에는 `docker.service` / `docker.socket` 이 실제로 올라왔는지 확인해야 합니다.
- GPU 컨테이너 실행은 `nvidia-ctk runtime configure --runtime=docker` 이후 다시 확인합니다.

Ubuntu 예시:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"${UBUNTU_CODENAME:-$VERSION_CODENAME}\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker buildx version
sudo systemctl status docker.socket --no-pager -l
sudo systemctl status docker.service --no-pager -l
docker run --rm hello-world
```

NVIDIA Container Toolkit 설정:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

최종 검증:

```bash
nvidia-smi
docker --version
docker buildx version
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

문제 해결 힌트:

> `docker: command not found` 가 뜨면 Docker Engine 이 아직 설치되지 않은 상태입니다.
> `docker: unknown command: docker buildx` 또는 `BuildKit is enabled but the buildx component is missing or broken`
> 가 뜨면 `docker-buildx-plugin`이 없는 상태입니다. `docker.io` 대신 Docker 공식 apt 저장소 기준 설치를 권장합니다.
> `could not select device driver "" with capabilities: [[gpu]]` 가 뜨면 NVIDIA Container Toolkit 미설치/미설정 상태일 가능성이 큽니다.
> `Cannot connect to the Docker daemon at unix:///var/run/docker.sock` 가 뜨면 daemon/socket 상태를 먼저 확인하세요:

```bash
sudo systemctl status docker.socket --no-pager -l
sudo systemctl status docker.service --no-pager -l
sudo systemctl daemon-reload
sudo systemctl reset-failed docker.service docker.socket
sudo systemctl enable docker.socket
sudo systemctl start docker.socket
sudo systemctl restart docker.service
docker version
```

### 3. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 `nvcr.io/nvidia/tensorrt` 베이스로 이미지를 만들고, `--gpus all` / `--runtime=nvidia`
중 동작하는 모드를 자동 감지해 실행 예시를 출력합니다. 베이스 이미지는
`./build.sh --base-image <image>` 또는 `TRT_BASE_IMAGE=... ./build.sh`로 바꿀 수 있습니다.
필요하면 `--torch-version`, `--torchvision-version`, `--trt-llm-version`으로 pin 값을 바꿀 수 있지만,
기본값은 `24.03` 계열과 맞춰 둔 값으로 두는 것을 권장합니다.

컨테이너 실행 예시:

```bash
docker run --gpus all -it --security-opt seccomp=unconfined \
  --name tensorrt-only \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:tensorrt
```

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
nvidia-smi || true
python3 -c "import unified_sdk; print('OK')"
python3 -c "import tensorrt as trt; from importlib import metadata; print('tensorrt=', getattr(trt, '__version__', metadata.version('tensorrt')))"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **NVIDIA GPU 가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `trt-only`
백엔드를 검증하는 표준 smoke 절차입니다.

```bash
# 1) 이미지 빌드
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
nvidia-smi || true
python3 -c "import tensorrt, pycuda; print('OK')"

# 4-a) 표준 fetching smoke (torchvision model zoo -> ONNX export -> .engine)
python3 examples/run_tensorrt_build.py \
  --model-name resnet50 \
  --precision fp32 \
  --input-name input \
  --input-shape 1,3,224,224

# 설치된 torchvision model zoo 이름 후보 확인
python3 examples/run_tensorrt_build.py --list-model-zoo

# 4-b) custom fetching smoke (provided .engine)
python3 examples/run_tensorrt_build.py \
  --engine /path/to/resnet50_FP32.engine \
  --model-name resnet50 \
  --precision fp32

# 4-c-1) ONNX → .engine 컴파일 (models/ 에서 자동 탐색)
python3 examples/run_tensorrt_build.py \
  --model-name yolov7 \
  --precision fp32 \
  --input-name images \
  --input-shape 1,3,640,640

# 4-c-2) PTH/PT -> ONNX export -> .engine 컴파일
#        (사용자 제공 checkpoint를 대상으로 하며, 여기서 pretrained weight를 새로 받지 않습니다)
python3 examples/run_tensorrt_build.py \
  --from-pth models/resnet50.pth \
  --model-name resnet50 \
  --precision fp32 \
  --input-name input \
  --input-shape 1,3,224,224

# 5) .engine 추론
python3 examples/run_tensorrt_infer.py \
  --engine-path build_output/yolov7_FP32.engine \
  --input-name images \
  --output-name output \
  --input-shape 1,3,640,640 \
  --iters 50

# 6) 엔진 입출력 메타 확인
python3 examples/inspect_engine_io.py build_output/yolov7_FP32.engine

# 7-a) (LLM) model id -> generate
python3 examples/run_tensorrt_llm_build.py \
  --model-ref TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode fetch

python3 examples/run_tensorrt_llm_infer.py \
  --engine-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"

# 7-b) (LLM) local model path + compatible prebuilt TensorRT-LLM artifact -> generate
python3 examples/run_tensorrt_llm_infer.py \
  --engine-path artifacts/tinyllama_trtllm \
  --tokenizer-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"

# 7-c) (LLM) local model path -> TensorRT-LLM compile -> generate
python3 examples/run_tensorrt_llm_build.py \
  --model-ref TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode llm_api_compile \
  --model-name tinyllama_trtllm \
  --max-model-len 512

python3 examples/run_tensorrt_llm_infer.py \
  --engine-path artifacts/tinyllama_trtllm \
  --tokenizer-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"

# 8) (LLM) artifact / model ref inspect
python3 examples/inspect_tensorrt_llm_model.py artifacts/tinyllama_trtllm --load
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 컴파일 (.engine 생성)

```python
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

cfg = BuildConfig(
    backend="tensorrt",
    model_or_path="models/yolov7.onnx",
    out_dir="build_output",
    model_name="yolov7",
    precision="fp32",                     # fp32 | fp16 | int8(calibrator 필요)
    input_name="images",
    min_input_shape=(1, 3, 640, 640),     # dynamic shape optimization profile
    opt_input_shape=(1, 3, 640, 640),
    max_input_shape=(1, 3, 640, 640),
    extra={"workspace_mib": 1024},
)
result = build_unified(cfg)
print(result.compiled_model_path)         # build_output/yolov7_FP32.engine
```

### LLM build / fetch

```python
from unified_sdk.types import LLMBuildConfig
from unified_sdk.build.api import build_unified_LLM

cfg = LLMBuildConfig(
    backend="tensorrt",
    model_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    out_dir="artifacts",
    model_name="tinyllama_trtllm",
    max_model_len=512,
    tensor_parallel_size=1,
    extra={"build_mode": "fetch"},        # fetch | llm_api_compile
)
result = build_unified_LLM(cfg)
print(result.compiled_model_path)
```

`run_tensorrt_build.py`는 현재 세 경로를 지원합니다.

- 표준 fetch: `torchvision` pretrained model fetch -> ONNX export -> TensorRT compile
- custom fetch: provided `.engine` copy/normalize
- custom compile:
  - `--onnx` 또는 `models/*.onnx`
  - `--from-pth` (`torchvision` model zoo 이름과 맞는 **사용자 제공** `.pth/.pt` checkpoint)

`--from-pth` 경로는:
- `--model-name`에 맞는 `torchvision` 아키텍처를 **구조 템플릿**으로 사용하고
- 실제 weight는 사용자가 제공한 checkpoint에서 읽어
- 네임스페이스 보정 후 ONNX export
- 마지막에 TensorRT compile
순으로 진행합니다.

### 추론

```python
import numpy as np
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime

cfg = RuntimeConfig(
    backend="tensorrt",
    engine_path="build_output/yolov7_FP32.engine",
    input_name="images",
    output_name="output",
    input_shape=(1, 3, 640, 640),
    use_execute_v3=True,                  # TRT 8.5+/10 권장 경로
)
rh = create_runtime(cfg)
y = infer(rh, np.zeros((1, 3, 640, 640), dtype=np.float32))
destroy_runtime(rh)
```

### LLM generate

```python
from unified_sdk.types import LLMRuntimeConfig
from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM, generate_LLM

cfg = LLMRuntimeConfig(
    backend="tensorrt",
    engine_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_model_len=512,
    max_tokens=32,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    tensor_parallel_size=1,
)
rh = create_runtime_LLM(cfg)
result = generate_LLM(rh, "What is the capital of South Korea?")
destroy_runtime_LLM(rh)
print(result)
```

## Runtime API 분리

`trt-only`는 runtime wrapping API를 vision과 LLM으로 구분합니다.

| 용도 | 단계 | Unified SDK | 내부 vendor |
| --- | --- | --- | --- |
| Vision `.engine` | 생성 | `create_runtime(cfg)` | `trt.Runtime(...).deserialize_cuda_engine(...)`, `engine.create_execution_context()` |
| Vision `.engine` | 추론 | `infer(rh, input_array)` | `execute_async_v3(...)` 또는 `execute_v2(...)` |
| Vision `.engine` | 종료 | `destroy_runtime(rh)` | device buffer `free()` 후 ctx clear |
| LLM / TensorRT-LLM | 빌드 | `build_unified_LLM(cfg)` | `tensorrt_llm.LLM(model=..., ...).save(engine_dir)` 또는 model ref pass-through |
| LLM / TensorRT-LLM | 생성 | `create_runtime_LLM(cfg)` | `tensorrt_llm.LLM(model=..., tokenizer=..., ...)` |
| LLM / TensorRT-LLM | 생성/추론 | `generate_LLM(rh, prompt, **overrides)` | `SamplingParams(...)`, `llm.generate(...)` |
| LLM / TensorRT-LLM | 종료 | `destroy_runtime_LLM(rh)` | `llm.shutdown/close/dispose` best-effort |

원칙:

- 기존 `create_runtime / infer / destroy_runtime`는 vision smoke 기준 API로 유지합니다.
- LLM 경로는 별도 `*_LLM` API를 통해 high-level generate path를 검증합니다.
- 내부 vendor surface가 달라도 Unified SDK 표면은 vision / LLM 용도별로 분리합니다.

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 NVIDIA TensorRT 위에서 동작하는 통합 추상화 계층이며, TensorRT/CUDA 의 라이선스 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 TensorRT 어댑터만 노출합니다. 다중 백엔드는 `main` 브랜치에서 사용하세요.
- TensorRT-LLM 경로는 high-level `generate` 중심 smoke를 제공합니다. 모델/옵션 호환성은 TensorRT-LLM 릴리스에 따라 달라질 수 있습니다.
- `tensorrt_llm`는 대형 wheel을 함께 끌어와 첫 빌드 시간이 길 수 있습니다. 대신 Dockerfile은 pip cache mount를 사용하므로, 같은 머신에서 이후 재빌드는 훨씬 덜 아프게 만드는 방향으로 정리했습니다.
- **Dynamic shape**: `min/opt/max_input_shape` 로 optimization profile 을 지정합니다.
  셋을 같은 값으로 주면 static shape 엔진이 됩니다.
- **정밀도**: `fp32` / `fp16` / `int8`. `int8` 은 calibrator 가 필수이며,
  `BuildConfig.extra["int8_calibrator"]` 없이 요청하면 **조용히 fp32 로 떨어지지 않고 명시적으로 실패**합니다.
- **실행 경로**: TRT 8.5+/10 은 `execute_async_v3` + `set_tensor_address`, 구버전은 `execute_v2` + bindings.
  `RuntimeConfig.use_execute_v3` 로 강제할 수 있고, 런타임이 지원 여부를 자동 감지합니다.
- **메모리**: device 버퍼(`cuda.mem_alloc`)는 `destroy_runtime()` 에서 명시적으로 `free()` 합니다.
- **lazy import**: `tensorrt`/`pycuda` 는 어댑터 내부에서만 import 하므로, GPU 없는 개발 환경에서도
  패키지 import 와 `--help` 가 동작합니다. `tensorrt_llm`도 LLM 어댑터 메서드 내부에서 lazy import 합니다.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_tensorrt_build.py --help`,
  `python3 examples/run_tensorrt_infer.py --help`, `python3 examples/inspect_engine_io.py --help`,
  `python3 examples/run_tensorrt_llm_build.py --help`, `python3 examples/run_tensorrt_llm_infer.py --help`,
  `python3 examples/inspect_tensorrt_llm_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-only`, `furiosa-llm-only`)에서 작업하세요.
