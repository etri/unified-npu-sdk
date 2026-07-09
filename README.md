# Unified SDK — TRT-only (NVIDIA TensorRT)

이 체크아웃(`trt-only` 브랜치)은 **NVIDIA TensorRT 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 TensorRT 1종으로 좁힌 버전입니다.

`rbln-only`·`qb-only`·`furiosa-only`와 동일한 단일-백엔드 패턴을 따릅니다.
컴파일은 **ONNX → `.engine`**(`Builder` + `OnnxParser` + `build_serialized_network`),
추론은 **TensorRT + PyCUDA**(`execute_async_v3` / `execute_v2`)를 사용합니다.

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
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (TensorRT 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── tensorrt_build.py       # TensorRT 빌드 어댑터
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / destroy_runtime
        ├── registry.py
        └── tensorrt_runtime.py     # TensorRT 런타임 어댑터 (PyCUDA)
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. `rbln-only`와 동일한 흐름(env → smoke → resnet50
> compile → infer)을 TensorRT 기준으로 구성했습니다.

---

## 💾 설치 방법

### 1. 호스트 사전 요구사항

- **NVIDIA GPU 드라이버**가 호스트에 정상 설치되어 있어야 합니다.
- **Docker Engine** 과 **NVIDIA Container Toolkit**이 함께 설치되어 있어야 Docker 컨테이너에서 GPU를 사용할 수 있습니다.
- `tensorrt`는 NVIDIA 공식 컨테이너(`nvcr.io/nvidia/tensorrt`)에 포함되어 있어 별도 설치가 필요 없습니다.
- 자세한 내용은 <https://developer.nvidia.com/tensorrt> 참조.

기본 확인:

```bash
nvidia-smi
docker --version
docker run --rm hello-world
```

`docker run --rm --gpus all ...` 에서 아래와 같은 에러가 뜨면:

```text
could not select device driver "" with capabilities: [[gpu]]
```

대개 **NVIDIA Container Toolkit 미설치/미설정** 상태입니다. Ubuntu 예시:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg2 docker.io

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

GPU 컨테이너 검증:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### 2. Docker 사전 준비

- `trt-only` 검증은 **Docker 기준**으로 진행합니다. 호스트에 `pip install -e .` 같은 로컬 직접 설치는 권장하지 않습니다.
- `docker` 명령이 없으면 먼저 Docker Engine 을 설치해야 합니다.
- `./build.sh`는 **BuildKit + buildx** 를 사용하므로 `docker buildx` 플러그인도 함께 준비되어 있어야 합니다.
- GPU 컨테이너 실행은 위 1번의 Toolkit 설정까지 끝난 뒤 확인합니다.

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
docker run --rm hello-world
```

GPU 컨테이너 사전 확인:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

> Ubuntu 기본 저장소의 `docker.io` 패키지만 설치한 경우 `docker buildx` 플러그인이 없을 수 있습니다.
> 이 경우 위 예시처럼 **Docker 공식 apt 저장소** 기준으로 `docker-buildx-plugin`까지 함께 설치하세요.
> `docker: command not found` 가 뜨면 Docker Engine 이 아직 설치되지 않은 상태입니다.
> `docker: unknown command: docker buildx` 또는 `BuildKit is enabled but the buildx component is missing or broken`
> 가 뜨면 `docker-buildx-plugin` 설치가 필요합니다.
> `--gpus all` 이 동작하지 않으면 NVIDIA Container Toolkit 설정을 먼저 완료해야 합니다.

### 3. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 `nvcr.io/nvidia/tensorrt` 베이스로 이미지를 만들고, `--gpus all` / `--runtime=nvidia`
중 동작하는 모드를 자동 감지해 실행 예시를 출력합니다. 베이스 이미지는
`./build.sh --base-image <image>` 또는 `TRT_BASE_IMAGE=... ./build.sh`로 바꿀 수 있습니다.

컨테이너 실행 예시:

```bash
docker run --gpus all -it --security-opt seccomp=unconfined \
  --name unified-sdk_trt_dev \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:trt
```

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
nvidia-smi || true
python3 -c "import unified_sdk; print('OK')"
python3 -c "import tensorrt; print('tensorrt=', tensorrt.__version__)"
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

# 4) ONNX → .engine 컴파일 (models/ 에서 자동 탐색)
python3 examples/run_tensorrt_build.py \
  --model-name yolov7 \
  --precision fp32 \
  --input-name images \
  --input-shape 1,3,640,640

# 5) .engine 추론
python3 examples/run_tensorrt_infer.py \
  --engine-path build_output/yolov7_FP32.engine \
  --input-name images \
  --output-name output \
  --input-shape 1,3,640,640 \
  --iters 50

# 6) 엔진 입출력 메타 확인
python3 examples/inspect_engine_io.py build_output/yolov7_FP32.engine
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

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 NVIDIA TensorRT 위에서 동작하는 통합 추상화 계층이며, TensorRT/CUDA 의 라이선스 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 TensorRT 어댑터만 노출합니다. 다중 백엔드는 `main` 브랜치에서 사용하세요.
- **Dynamic shape**: `min/opt/max_input_shape` 로 optimization profile 을 지정합니다.
  셋을 같은 값으로 주면 static shape 엔진이 됩니다.
- **정밀도**: `fp32` / `fp16` / `int8`. `int8` 은 calibrator 가 필수이며,
  `BuildConfig.extra["int8_calibrator"]` 없이 요청하면 **조용히 fp32 로 떨어지지 않고 명시적으로 실패**합니다.
- **실행 경로**: TRT 8.5+/10 은 `execute_async_v3` + `set_tensor_address`, 구버전은 `execute_v2` + bindings.
  `RuntimeConfig.use_execute_v3` 로 강제할 수 있고, 런타임이 지원 여부를 자동 감지합니다.
- **메모리**: device 버퍼(`cuda.mem_alloc`)는 `destroy_runtime()` 에서 명시적으로 `free()` 합니다.
- **lazy import**: `tensorrt`/`pycuda` 는 어댑터 내부에서만 import 하므로, GPU 없는 개발 환경에서도
  패키지 import 와 `--help` 가 동작합니다.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_tensorrt_build.py --help`,
  `python3 examples/run_tensorrt_infer.py --help`, `python3 examples/inspect_engine_io.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-only`, `furiosa-llm-only`)에서 작업하세요.
