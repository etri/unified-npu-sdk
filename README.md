# Unified SDK — TRT-only (NVIDIA TensorRT)

이 체크아웃(`trt-only` 브랜치)은 **NVIDIA TensorRT 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 TensorRT 1종으로 좁힌 버전입니다.

`rbln-only`·`qb-only`·`furiosa-only`와 동일한 단일-백엔드 패턴을 따릅니다.
컴파일은 **ONNX → `.engine`**(`Builder` + `OnnxParser` + `build_serialized_network`),
추론은 **TensorRT + PyCUDA**(`execute_async_v3` / `execute_v2`)를 사용합니다.
LLM 경로는 **TensorRT-LLM**(`tensorrt_llm.LLM`, `SamplingParams`, `llm.generate`)를 사용합니다.
이 브랜치는 **브랜치는 하나(`trt-only`)로 유지**하되, Docker 환경은 **vision / llm 두 flavor**로 분리합니다.

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 TensorRT 단일 백엔드 분기입니다.
TensorRT 분기는 국산 NPU 백엔드들의 **비교 기준(reference)** 역할을 합니다.

### 현재 구현 상태

| 구분 | 현재 상태 |
| --- | --- |
| Vision API | `build_unified` / `create_runtime` / `infer` / `destroy_runtime` 구현 |
| LLM API | `build_unified_LLM` / `create_runtime_LLM` / `infer_LLM` / `generate_LLM` / `destroy_runtime_LLM` 구현 |
| Vision smoke | 표준 fetch / provided `.engine` fetch / ONNX compile / PTH->ONNX->`.engine` / infer / inspect 구현 |
| LLM smoke | `7-a` model id fetch, `7-b` local path/artifact fetch, `7-c` custom compile, `8` infer/inspect 흐름 구현 |

### 주요 이슈

- Docker 환경은 `vision` / `llm` flavor 로 분리해 유지합니다.
- `llm` flavor 는 official TensorRT-LLM release container(PyTorch backend) 기준입니다.
- LLM custom compile 의 기본 smoke 경로는 `local HF path -> checkpoint prepare -> trtllm-build CLI compile` 입니다.
- `llm` 이미지는 matching TensorRT-LLM source checkout fallback 을 함께 포함해, checkpoint prepare wrapper 를 self-contained 하게 유지합니다.
- 로컬 prebuilt artifact fetch(`7-b`)는 compile과 독립 경로입니다.

---

## 🏗️ 프로젝트 구조

```
<repo-root>/
├── README.md
├── LICENSE
├── pyproject.toml
├── pyrightconfig.json
├── Dockers/
│   ├── docker.trt.vision
│   ├── docker.trt.llm
│   ├── requirements.trt.shared.txt
│   ├── requirements.trt.vision.txt
│   └── requirements.trt.llm.txt
├── devcontainer.json
├── build.sh
├── scripts/
│   └── build_trt.sh
├── examples/
│   ├── run_tensorrt_build.py       # ONNX → .engine 컴파일
│   ├── run_tensorrt_infer.py       # .engine 추론 + latency 측정
│   └── inspect_engine_io.py        # .engine 입출력 텐서 메타 확인
│   ├── run_tensorrt_llm_build.py   # model ref/path -> TensorRT-LLM fetch/compile
│   ├── run_tensorrt_llm_infer.py   # TensorRT-LLM generate
│   ├── inspect_tensorrt_llm_model.py # TensorRT-LLM artifact/model ref 점검
│   └── llama/convert_checkpoint.py # TensorRT-LLM llama/tinyllama checkpoint 준비 wrapper
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (typed backend_options / prepared_input)
    ├── options.py                  # TensorRT typed backend options
    ├── frontends/
    │   ├── __init__.py             # vision/llm build request resolve/prepare
    │   └── types.py
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
- `trt-only`는 Docker를 두 flavor로 나눕니다.
  - `vision`: 일반 TensorRT `.engine` build/infer 전용
  - `llm`: TensorRT-LLM generate/build 전용
- `vision` flavor 기본 base image는 `nvcr.io/nvidia/tensorrt:24.03-py3`입니다. `vision`은 TensorRT Python import 안정성을 우선하고, 필요한 `torch` / `torchvision`만 별도로 올립니다.
- `llm` flavor 기본 base image는 `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22`입니다. TensorRT-LLM은 수동 pip 조합보다 공식 release container 축이 더 안정적이어서, LLM Docker는 이쪽을 기본으로 둡니다.
- `llm` flavor 이미지는 release container 위에 matching TensorRT-LLM source checkout(`v1.3.0rc22` 기본)을 함께 포함합니다. checkpoint prepare wrapper 는 installed API 를 우선 쓰고, 필요하면 이 bundled source 를 fallback 으로 사용합니다.
- 2026년 7월 25일 기준 `vision` flavor는 TensorRT가 이미 포함된 base image를 사용하고, `torch==2.2.2`, `torchvision==0.17.2`만 별도 설치합니다.
- `llm` flavor는 official TensorRT-LLM release container를 기준으로 하고, Unified SDK public LLM API는 유지한 채 내부 vendor mapping만 그 컨테이너가 제공하는 TensorRT-LLM API 축에 맞춰 씁니다.
- 최신 TensorRT-LLM 1.x release container에서는 PyTorch backend가 기본이며, Unified SDK의 `max_model_len`은 내부 vendor 호출 시 `max_seq_len`으로 매핑합니다.
- 이전에는 `pytorch` base 위에 `pip tensorrt`를 올리는 방식을 시도했지만, 설치가 끝나도 `import tensorrt`가 실패하는 경우가 있어 `vision` flavor는 TensorRT base로 되돌렸습니다.
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

기본 원칙:

- `vision smoke`는 `--flavor vision`
- `llm smoke`는 `--flavor llm`
- 기본 컨테이너 이름도 각각 `trt-only-vision`, `trt-only-llm` 으로 분리됩니다.

```bash
./build.sh --flavor vision
./build.sh --flavor llm
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 flavor에 따라 다른 Dockerfile을 사용합니다.

- `--flavor vision`
  - Dockerfile: `Dockers/docker.trt.vision`
  - image tag: `unified-sdk:tensorrt-vision`
  - container name: `trt-only-vision`
- `--flavor llm`
  - Dockerfile: `Dockers/docker.trt.llm`
  - image tag: `unified-sdk:tensorrt-llm`
  - container name: `trt-only-llm`

베이스 이미지는 `./build.sh --flavor <...> --base-image <image>`로 바꿀 수 있습니다.
vision 컨테이너 실행 예시:

```bash
docker run --gpus all -it --security-opt seccomp=unconfined \
  --name trt-only-vision \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:tensorrt-vision
```

llm 컨테이너 실행 예시:

```bash
docker run --gpus all -it --security-opt seccomp=unconfined \
  --name trt-only-llm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:tensorrt-llm
```

vision 컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
nvidia-smi || true
python3 -c "import unified_sdk; print('OK')"
python3 -c "import tensorrt as trt; from importlib import metadata; print('tensorrt=', getattr(trt, '__version__', metadata.version('tensorrt')))"
```

llm 컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
nvidia-smi || true
python3 -c "import unified_sdk; print('OK')"
python3 -c "import tensorrt_llm; print('tensorrt_llm OK')"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **NVIDIA GPU 가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `trt-only`
백엔드를 검증하는 표준 smoke 절차입니다. `vision`과 `llm`은 **서로 다른 Docker flavor**에서
검증합니다.

### Vision smoke

`vision` smoke는 `trt-only-vision` 컨테이너에서 진행합니다.

```bash
# 1) vision 이미지 빌드
./build.sh --flavor vision

# 2) build.sh가 출력한 docker run 명령으로 vision 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
nvidia-smi || true
python3 -c "import tensorrt, pycuda, torch, torchvision; print('vision stack OK')"

# 4-a) 표준 fetching smoke (torchvision model zoo -> ONNX export -> .engine)
python3 examples/run_tensorrt_build.py \
  --model-name resnet50 \
  --precision fp32 \
  --input-name input \
  --input-shape 1,3,224,224

# 설치된 torchvision model zoo 이름 후보 확인
python3 examples/run_tensorrt_build.py --list-model-zoo

# 4-b) custom fetching smoke (provided .engine)
#      예: 표준 fetch 결과물을 다시 넣어볼 때는 ./build_output/resnet50_FP32.engine
python3 examples/run_tensorrt_build.py \
  --engine /path/to/resnet50_FP32.engine \
  --model-name resnet50 \
  --precision fp32

# 참고: provided .engine 도 출력 산출물은 요청 precision 기준으로
# build_output/<model-name>_<PREC>.engine 로 normalize 됩니다.

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
```

### LLM smoke

`LLM` smoke는 `trt-only-llm` 컨테이너에서 진행합니다.

진행 원칙:

- `fetch`, `custom_compile`, `runtime(generate)`는 phase를 구분해서 봅니다.
- `fetch`는 artifact 생성이 아니라, `model id / local HF path / local artifact dir` 중 무엇을 runtime 입력으로 쓸지 계약을 확정하는 단계입니다.
- `custom_compile`은 실제 TensorRT-LLM artifact dir를 만드는 단계입니다.
- `runtime(generate)`는 실행 표면입니다. 다만 TensorRT-LLM vendor runtime 특성상 `model id`나 `local HF path`를 직접 주면 내부 load/build-like 동작이 다시 보일 수 있습니다.
- `7-a`는 `model id -> fetch -> generate` 기본 경로입니다.
- `7-b`는 `local HF path`를 직접 쓰는 경로입니다.
- 따라서 `7-b`를 돌리기 전에는 먼저 TinyLlama 같은 모델을 로컬 경로 아래에 준비해야 합니다.
- `7-c`는 `custom_compile` 경로입니다.
  스모크 기본 경로는 `local HF path -> TRT-LLM checkpoint prepare -> trtllm-build -> artifact runtime` 흐름입니다.
  `llm` 이미지는 matching TensorRT-LLM source fallback 을 포함하므로, checkpoint prepare 단계도 가능한 한 self-contained 하게 따라갈 수 있습니다.
  다만 checkpoint prepare 단계는 여전히 설치된 TensorRT-LLM API surface와 모델 지원성의 영향을 받습니다.

```bash
# 1) llm 이미지 빌드
./build.sh --flavor llm

# 2) build.sh가 출력한 docker run 명령으로 llm 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
nvidia-smi || true
python3 -c "import tensorrt_llm; print('tensorrt_llm OK')"

# 7-a) (LLM) model id -> generate
python3 examples/run_tensorrt_llm_build.py \
  --model-ref TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode fetch

python3 examples/run_tensorrt_llm_infer.py \
  --engine-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"

# 7-b-1) (LLM) local HF path 준비
#        예: TinyLlama repo snapshot 을 ./models 아래로 준비
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ./models/TinyLlama-1.1B-Chat-v1.0

# 7-b-2) (LLM) local HF path -> fetch -> generate
python3 examples/run_tensorrt_llm_build.py \
  --model-ref ./models/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode fetch

python3 examples/run_tensorrt_llm_infer.py \
  --engine-path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"

# 메모) prebuilt TensorRT-LLM artifact dir 가 이미 있으면 local HF path 대신 바로 runtime 입력으로 사용할 수 있습니다.
#       artifact dir 는 아래 7-c 결과물이거나, vendor가 미리 생성한 artifact dir 여도 됩니다.
# python3 examples/run_tensorrt_llm_infer.py \
#   --engine-path artifacts/tinyllama_trtllm \
#   --tokenizer-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#   --prompt "What is the capital of South Korea?"

# 7-c-1) (LLM) local HF path -> TensorRT-LLM checkpoint dir 준비
#        같은 TinyLlama local HF path 를 재사용합니다.
#        unified-sdk/examples wrapper 는 installed TensorRT-LLM conversion API를 우선 사용하고,
#        필요하면 llm 이미지에 포함된 matching TensorRT-LLM source fallback 을 사용합니다.
python3 examples/llama/convert_checkpoint.py \
  --model_dir ./models/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir ./models/tinyllama_trtllm_ckpt \
  --dtype float16

# 7-c-2) (LLM) local TensorRT-LLM checkpoint dir -> custom compile via trtllm-build
python3 examples/run_tensorrt_llm_build.py \
  --model-ref ./models/tinyllama_trtllm_ckpt \
  --build-mode custom_compile \
  --model-name tinyllama_trtllm \
  --max-model-len 512

# 7-c-3) (LLM) compiled artifact -> generate
python3 examples/run_tensorrt_llm_infer.py \
  --engine-path artifacts/tinyllama_trtllm \
  --tokenizer-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"

# 8) (LLM) artifact / model ref inspect
python3 examples/inspect_tensorrt_llm_model.py artifacts/tinyllama_trtllm --load
```

`run_tensorrt_llm_infer.py`는 기본적으로 `--chat-template auto` 동작을 사용합니다.
즉 TinyLlama 같은 chat/instruct 모델에 대해 tokenizer를 찾을 수 있으면 `apply_chat_template(...)`
를 자동 시도하고, `tokenizer_ref`, `formatted_prompt`를 함께 출력해 실제 입력 프롬프트를 확인할 수 있습니다.

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 컴파일 (.engine 생성)

```python
from unified_sdk.build.api import build_unified
from unified_sdk.frontends import resolve_tensorrt_vision_build_request
from unified_sdk.frontends.types import TensorRTVisionFrontendBuildRequest
from unified_sdk.options import TensorRTVisionBuildOptions
from unified_sdk.types import BuildConfig

resolved = resolve_tensorrt_vision_build_request(
    TensorRTVisionFrontendBuildRequest(
        model_name="yolov7",
        models_dir=Path("models"),
        out_dir=Path("build_output"),
        onnx_path=Path("models/yolov7.onnx"),
        input_name="images",
        min_input_shape=(1, 3, 640, 640),
        opt_input_shape=(1, 3, 640, 640),
        max_input_shape=(1, 3, 640, 640),
    )
)

cfg = BuildConfig(
    backend="tensorrt",
    model_or_path=resolved.model_or_path,
    out_dir="build_output",
    model_name="yolov7",
    backend_options=TensorRTVisionBuildOptions(
        precision="fp32",                 # fp32 | fp16 | int8(calibrator 필요)
        workspace_mib=1024,
    ),
    prepared_input=resolved.prepared_input,
)
result = build_unified(cfg)
print(result.compiled_model_path)         # build_output/yolov7_FP32.engine
```

### LLM build / fetch

```python
from unified_sdk.build.api import build_unified_LLM
from unified_sdk.frontends import resolve_tensorrt_llm_build_request
from unified_sdk.frontends.types import TensorRTLLMFrontendBuildRequest
from unified_sdk.options import TensorRTLLMBuildOptions
from unified_sdk.types import LLMBuildConfig

resolved = resolve_tensorrt_llm_build_request(
    TensorRTLLMFrontendBuildRequest(
        model_ref="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        out_dir=Path("artifacts"),
        model_name="tinyllama_trtllm",
        build_mode="fetch",
    )
)

cfg = LLMBuildConfig(
    backend="tensorrt",
    model_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    out_dir="artifacts",
    model_name="tinyllama_trtllm",
    backend_options=TensorRTLLMBuildOptions(
        build_mode="fetch",
        max_model_len=512,
        tensor_parallel_size=1,
    ),
    prepared_input=resolved.prepared_input,
)
result = build_unified_LLM(cfg)
print(result.compiled_model_path)
```

LLM build phase는 이렇게 읽으면 됩니다.

- `fetch`: model id, local HF path, local prebuilt TensorRT-LLM artifact dir
- `custom_compile`: local HF path -> checkpoint prepare -> local TensorRT-LLM checkpoint dir -> `trtllm-build`

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
from unified_sdk.runtime import create_runtime, infer, destroy_runtime
from unified_sdk.options import TensorRTVisionRuntimeOptions
from unified_sdk.types import RuntimeConfig

cfg = RuntimeConfig(
    backend="tensorrt",
    engine_path="build_output/yolov7_FP32.engine",
    input_name="images",
    output_name="output",
    input_shape=(1, 3, 640, 640),
    backend_options=TensorRTVisionRuntimeOptions(
        use_execute_v3=True,              # TRT 8.5+/10 권장 경로
    ),
)
rh = create_runtime(cfg)
y = infer(rh, np.zeros((1, 3, 640, 640), dtype=np.float32))
destroy_runtime(rh)
```

### LLM generate

```python
from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM, generate_LLM
from unified_sdk.options import TensorRTLLMRuntimeOptions
from unified_sdk.types import LLMRuntimeConfig

cfg = LLMRuntimeConfig(
    backend="tensorrt",
    model_ref_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_tokens=32,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    backend_options=TensorRTLLMRuntimeOptions(
        max_model_len=512,
        tensor_parallel_size=1,
    ),
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
| LLM / TensorRT-LLM | 빌드 | `build_unified_LLM(cfg)` | model ref/local path pass-through 또는 checkpoint prepare 후 `trtllm-build --checkpoint_dir ... --output_dir ...` |
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
- `trt-only`는 branch 하나를 유지하되 Docker flavor를 둘로 분리합니다. vision과 llm은 같은 Unified SDK public API 구조를 공유하지만, vendor stack mismatch를 줄이기 위해 컨테이너를 분리합니다.
- `vision` flavor는 일반 TensorRT Python stack을 직접 설치하고, `llm` flavor는 official TensorRT-LLM release container를 기본으로 씁니다.
- 소스 코드는 이미지에 bake하지 않고 bind mount 기준으로 동작하게 해 두었기 때문에, 코드 수정만으로는 환경 레이어를 다시 만들지 않도록 정리했습니다.
- `llm` flavor에서 `7-b`는 실제 로컬 artifact dir가 있을 때만 동작합니다. 경로가 없으면 HF repo id로 오인하지 않도록 로컬 경로 missing 에러를 먼저 냅니다.
- `llm` flavor에서 local HF path custom compile 은 checkpoint prepare 후 `trtllm-build` 로 이어지고, local checkpoint dir 는 곧바로 `trtllm-build` 입력이 됩니다.
- **Dynamic shape**: `min/opt/max_input_shape` 로 optimization profile 을 지정합니다.
  셋을 같은 값으로 주면 static shape 엔진이 됩니다.
  현재 `allow_dynamic_shape=True`는 입력 shape 변경 시 runtime rebind/reallocation까지 수행합니다.
  다만 smoke 기준은 여전히 먼저 fixed-shape baseline을 통과한 뒤, dynamic profile 엔진에서 추가 검증하는 순서를 권장합니다.
- **정밀도**: `fp32` / `fp16` / `int8`. `int8` 은 calibrator 가 필수이며,
  `TensorRTVisionBuildOptions(int8_calibrator=...)` 없이 요청하면 **조용히 fp32 로 떨어지지 않고 명시적으로 실패**합니다.
- **실행 경로**: TRT 8.5+/10 은 `execute_async_v3` + `set_tensor_address`, 구버전은 `execute_v2` + bindings.
  `TensorRTVisionRuntimeOptions(use_execute_v3=...)` 로 제어할 수 있고, 런타임이 지원 여부를 자동 감지합니다.
- **메모리**: device 버퍼(`cuda.mem_alloc`)는 `destroy_runtime()` 에서 명시적으로 `free()` 합니다.
- **lazy import**: `tensorrt`/`pycuda` 는 어댑터 내부에서만 import 하므로, GPU 없는 개발 환경에서도
  패키지 import 와 `--help` 가 동작합니다. `tensorrt_llm`도 LLM 어댑터 메서드 내부에서 lazy import 합니다.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_tensorrt_build.py --help`,
  `python3 examples/run_tensorrt_infer.py --help`, `python3 examples/inspect_engine_io.py --help`,
  `python3 examples/run_tensorrt_llm_build.py --help`, `python3 examples/run_tensorrt_llm_infer.py --help`,
  `python3 examples/inspect_tensorrt_llm_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-only`, `furiosa-llm-only`)에서 작업하세요.
