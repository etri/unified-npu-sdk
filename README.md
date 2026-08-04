# Unified SDK — QB-only (Mobilint ARISE)

이 체크아웃(`qb-only` 브랜치)은 **Mobilint ARISE(QB) NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 QB 1종으로 좁힌 버전입니다.

이 브랜치는 `main`과 같은 Unified SDK 표면을 유지하되, 실제 구현은 `rbln-only`·`trt-only`처럼 QB 백엔드 하나에만 집중한 단일-백엔드 분기입니다.
모델 컴파일은 Mobilint compiler Python API(`qubee` 또는 `qbcompiler`로 노출될 수 있음)로 수행하고, 추론은 `qbruntime`을 사용합니다. 기본 개발/검증 경로는 Mobilint 공식 문서 흐름에 맞춰 **공식 `qbcompiler` Docker 이미지**, **벤더 제공 `qbcompiler` wheel**, **`pip install mobilint-qb-runtime`**, **Mobilint APT 저장소의 `mobilint-cli`** 조합을 기준으로 정리했습니다.

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 QB(Mobilint ARISE) 단일 백엔드 분기입니다.

### 현재 구현 상태

| 구분 | 현재 상태 |
| --- | --- |
| Vision API | `build_unified` / `create_runtime` / `infer` / `destroy_runtime` 지원 |
| Sequence low-level API | `create_sequence_runtime` / `infer_sequence` / `destroy_sequence_runtime` 지원 |
| Vision compile | 표준 fetch, 사전 컴파일된 `.mxq` fetch, ONNX compile, `resnet50` 기준 PTH->ONNX->`.mxq` 경로 지원 |
| LLM compile | precompiled `.mxq` fetch 및 low-level runtime smoke 중심, custom compile 은 `planned` |

### 주요 이슈

- 현재 공개 Mobilint 문서 기준으로는 **local LLM source/checkpoint -> qb compiler -> `.mxq`** 경로를 branch public API로 일반화하기 어려워 `build_unified_LLM(cfg)`는 `planned` 상태로 남겨둡니다.
- 현재 LLM 지원 완료 기준은 high-level generate 가 아니라 **low-level cache-aware infer smoke 통과**입니다.

---

## 🏗️ 프로젝트 구조

```
<repo-root>/
├── README.md
├── LICENSE
├── pyproject.toml
├── pyrightconfig.json
├── Dockers/
│   ├── docker.qb.unified
│   └── requirements.qb.unified.txt
├── devcontainer.json
├── build.sh
├── scripts/
│   └── build_qb.sh
├── vendor/                         # (gitignore) Mobilint compiler wheel 배치 위치
│   └── README.md                   #   qbcompiler-*.whl
├── examples/
│   ├── run_qb_build.py             # .mxq 확보(fetch) 또는 ONNX→.mxq 컴파일(compiler Python API)
│   ├── run_qb_infer.py             # .mxq 모델 추론 (qbruntime)
│   ├── inspect_qb_model.py         # .mxq 요약 정보 확인
│   ├── prepare_qb_transformer_model.py  # Mobilint HF group 의 transformer/LLM MXQ snapshot 준비
│   ├── run_qb_llm_infer.py         # low-level cache-aware LLM infer smoke
│   └── inspect_qb_llm_model.py     # LLM MXQ의 cache/meta 정보 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (QB 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── qb_build.py             # QB 빌드 어댑터 (qubee/qbcompiler mxq_compile)
    ├── runtime/
    │   ├── __init__.py
    │   ├── api.py                  # vision: create_runtime/infer/destroy_runtime
    │   ├── registry.py
    │   └── qb_runtime.py           # QB vision 런타임 어댑터 (qbruntime)
    └── sequence_runtime/
        ├── __init__.py
        ├── api.py                  # low-level sequence runtime: create_sequence_runtime/infer_sequence/destroy_sequence_runtime
        ├── registry.py
        └── qb_sequence_runtime.py  # QB cache-aware sequence runtime 어댑터
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. `rbln-only`와 동일한 흐름(env → smoke → resnet50
> compile → infer)을 compiler Python API(qubee/qbcompiler) / qbruntime / mobilint-cli 기준으로 구성했습니다.

### Runtime API 분리

`qb-only`는 runtime wrapping API를 **vision**과 **sequence low-level runtime**으로 구분하며,
실제로는 아래처럼 `qbruntime` 함수에 매핑됩니다.

| 용도 | 단계 | Unified SDK | 내부 vendor |
| --- | --- | --- | --- |
| Vision `.mxq` | 생성 | `create_runtime(cfg)` | `qbruntime.model.load(...)` |
| Vision `.mxq` | 추론 | `infer(rh, input_array)` | `model.infer([input_array])` |
| Vision `.mxq` | 종료 | `destroy_runtime(rh)` | `model.dispose/release/unload/close` |
| Sequence / Transformer `.mxq` | 생성 | `create_sequence_runtime(cfg)` | `qbruntime.model.load(...)` |
| Sequence / Transformer `.mxq` | 추론 | `infer_sequence(rh, input_array, cache_size=..., batch_params=...)` | `model.infer([input_array], cache_size=..., params=...)` |
| Sequence / Transformer `.mxq` | 종료 | `destroy_sequence_runtime(rh)` | `model.dispose/release/unload/close` |

기본 원칙:
- 기존 `create_runtime / infer / destroy_runtime`는 vision smoke 기준 API로 유지합니다.
- low-level sequence preview는 별도 `sequence_runtime` capability를 통해 cache-aware runtime path를 검증합니다.
- 내부 vendor runtime은 모두 `qbruntime`이지만, Unified SDK 표면은 용도별로 분리합니다.

---

## 💾 설치 방법

### 1. 저장소 체크아웃 & 컴파일러 wheel 배치

이 브랜치는 두 방식 모두 지원합니다.

- 별도 worktree 폴더 예: `.../qb-only/`
- 일반 저장소 루트 예: `.../unified-npu-sdk/`에서 `git switch qb-only`

Mobilint 공식 문서 기준으로 SDK qb는 `Driver / qb Runtime / qb Compiler`로 나뉩니다. 이 브랜치는 아래 조합을 기본 경로로 사용합니다.

- **Compiler base**: Mobilint 공식 `qbcompiler` Docker 이미지
- **Compiler Python API**: 벤더 제공 `qbcompiler-*.whl` (`qubee` 또는 `qbcompiler` import로 노출될 수 있음)
- **Runtime**: `pip install mobilint-qb-runtime`
- **CLI Utility**: Mobilint APT 저장소 등록 후 `apt install mobilint-cli`

따라서 `vendor/`에는 **`qbcompiler` compiler wheel만** 두는 것을 기준으로 설명합니다. 패키지 버전에 따라 compiler Python import 이름은 `qubee` 또는 `qbcompiler`일 수 있습니다.

> 권장: `vendor/`에는 `qbcompiler` wheel을 **한 버전만** 두세요. 여러 버전을 같이 두면 어떤 wheel 기준으로
> base image를 추론할지 헷갈릴 수 있으므로, 테스트에 사용할 버전 하나만 남기는 것이 안전합니다.
> 현재 `./build.sh`도 여러 버전이 있으면 그대로 진행하지 않고, 하나만 남기거나 `--compiler-wheel <filename>`로
> 명시 선택하라고 에러를 냅니다.

```bash
# 예시 1) 별도 worktree
# cd ~/Codings/Micro_DC/qb-only

# Mobilint compiler wheel 배치 (docs.mobilint.com 참조)
cp /path/to/qbcompiler-*.whl vendor/

# 여러 버전이 같이 있으면 명시 선택 가능
./build.sh --compiler-wheel qbcompiler-1.1.2+aries2-py3-none-any.whl
```

### 2. Docker 사전 준비

- `qb-only` 검증은 **Docker 기준**으로 진행합니다.
- Mobilint 공식 compiler 설치 문서는 **버전 태그가 붙은** `qbcompiler` Docker 이미지를 기준으로 설명합니다. 이 브랜치도 같은 방향을 따르며, 기본 베이스 이미지는 `vendor/qbcompiler-*.whl`의 버전에서 자동 추론한 `mobilint/qbcompiler:<major>.<minor>-cpu-ubuntu22.04` 입니다.
  - 예: `qbcompiler-1.1.2+aries2-py3-none-any.whl` -> `mobilint/qbcompiler:1.1-cpu-ubuntu22.04`
  - 예: `qbcompiler-1.2.0-py3-none-any.whl` -> `mobilint/qbcompiler:1.2-cpu-ubuntu22.04`
  - wheel 이름에서 버전을 추론할 수 없으면 `--base-image`로 직접 지정해야 합니다.
- 실제로 pull 가능한 compiler 태그는 Mobilint Docker Hub tags 페이지에서 먼저 확인하는 것을 권장합니다:
  <https://hub.docker.com/r/mobilint/qbcompiler/tags>
- Mobilint Docker Hub 태그 기준 compiler 이미지는 **CPU 전용**(`mobilint/qbcompiler:<major>.<minor>-cpu-ubuntu22.04`)과 **GPU 가속**(`mobilint/qbcompiler:<major>.<minor>-cuda12.8.1-ubuntu22.04`) 두 종류가 있습니다.
  - 일반적인 wrapper smoke나 CPU-only compile이면 `-cpu` 이미지를 쓰면 됩니다.
  - GPU 가속 compile이 필요한 환경에서만 `--base-image mobilint/qbcompiler:<major>.<minor>-cuda12.8.1-ubuntu22.04`처럼 명시적으로 바꿔 쓰세요.
- Ubuntu에서는 **Docker 공식 apt 저장소** 기준 설치를 권장합니다. `docker.io`만 설치하면 `docker buildx`가 없을 수 있습니다.
- `./build.sh`를 돌리기 전에 `docker.service` / `docker.socket` 이 실제로 올라왔는지 확인하세요.
- `qb-only` Docker 정의는 `Dockers/docker.qb.unified`에 있으며, 컨테이너 안에서 `mobilint-cli`를 설치하기 위해 Mobilint APT 저장소(`https://dl.mobilint.com/apt`)를 추가합니다.

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

자주 겪는 문제:

> `docker: unknown command: docker buildx` 또는 `BuildKit is enabled but the buildx component is missing or broken`
> 가 뜨면 `docker-buildx-plugin`이 없는 상태입니다.
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

### 3. 호스트 사전 요구사항

- **Mobilint ARISE 드라이버**가 호스트에 설치되어 있어야 합니다 (`mobilint-cli status`로 확인).
  자세한 절차는 <https://docs.mobilint.com/v1.3/en/introduction.html> 및
  <https://docs.mobilint.com/v1.3/en/installing_runtime_library.html> 참조.
- 컨테이너 실행 시 실제로 존재하는 장치 노드(`/dev/aries*` 또는 `/dev/arise*`)만 `--device`로 전달합니다.
- 표준 model zoo fetch helper는 참조 검증 기준 경로(`~/.mblt_model_zoo/vision/<product>/<core_mode>`)를 찾기 위해
  기본 `core_mode=global8`을 사용합니다.
- 반면 `.mxq` 추론 helper(`run_qb_infer.py`)는 로컬/직접 컴파일된 산출물이 `Single`만 지원하는 경우가 있어
  기본 `core_mode=auto`를 사용합니다. 필요할 때만 `--core-mode global4|global8|single`로 고정하세요.

### 4. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 기본적으로 `torch`/`torchvision`을 CPU wheel index
(`https://download.pytorch.org/whl/cpu`)에서 설치하고, 다음 조합으로 이미지를 구성합니다.

- base image: `mobilint/qbcompiler:<major>.<minor>-cpu-ubuntu22.04` (기본은 `qbcompiler-*.whl` 버전에서 자동 추론)
- compiler wheel: `vendor/qbcompiler-*.whl`
- runtime pip package: `mobilint-qb-runtime`

CPU / GPU 예시:

```bash
# CPU-only compiler image (default)
./build.sh --base-image mobilint/qbcompiler:1.2-cpu-ubuntu22.04

# GPU-accelerated compiler image
./build.sh --base-image mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04
```

다른 값을 쓰려면:

```bash
QB_BASE_IMAGE=... ./build.sh
QB_RUNTIME_PIP_SPEC=... ./build.sh
PYTORCH_INDEX_URL=... ./build.sh
```

또는:

```bash
./build.sh --base-image <image> --runtime-pip-spec <pip-spec> --pytorch-index-url <url>
```

컨테이너 실행 예시:

```bash
docker run -it --security-opt seccomp=unconfined \
  --name qb-only \
  --device /dev/aries0:/dev/aries0 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:qb
```

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
command -v mobilint-cli && mobilint-cli status || true
python3 -c "import unified_sdk, qbruntime; print('OK')"
python3 -c "import mblt_model_zoo; print('mblt_model_zoo=', getattr(mblt_model_zoo, '__version__', 'unknown'))"
python3 -c "import importlib, pkgutil; m = next((importlib.import_module(n) for n in ('qubee', 'qbcompiler') if pkgutil.find_loader(n)), None); print('compiler_pkg=', getattr(m, '__name__', 'missing'))"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **Mobilint ARISE 장치가 호스트에 연결된 단일 머신**에서 Docker로 `qb-only`
백엔드를 검증하는 표준 smoke 절차입니다. 이 경로에서는 별도 중간 래퍼 없이 Unified SDK의 QB adapter가
vendor SDK(compiler Python API `qubee`/`qbcompiler`, runtime `qbruntime`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드 (vendor/ 에 qbcompiler wheel 필요, runtime 은 pip 설치)
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
command -v mobilint-cli && mobilint-cli status || true
python3 -c "import unified_sdk, qbruntime; print('OK')"
python3 -c "import qbruntime; from qbruntime import type as t; print('devices=', t.get_available_device_numbers())"
python3 -c "import mblt_model_zoo; print('mblt_model_zoo=', getattr(mblt_model_zoo, '__version__', 'unknown'))"
python3 -c "import importlib, pkgutil; m = next((importlib.import_module(n) for n in ('qubee', 'qbcompiler') if pkgutil.find_loader(n)), None); print('compiler_pkg=', getattr(m, '__name__', 'missing'), 'version=', getattr(m, '__version__', 'unknown') if m else 'n/a')"
```

### 4-a) 표준 fetching smoke

Mobilint 공식 문서 흐름대로 이미 생성된 `.mxq`가 `~/.mblt_model_zoo/...` 아래에 있으면,
`run_qb_build.py --model-name ...`가 해당 경로를 자동 탐색해 fetch 합니다.
없으면 `mblt_model_zoo.vision.*` 심볼에서 `--model-name`과 이름이 맞는 클래스를 찾아
1회 실행해 `.mxq`를 materialize 한 뒤 다시 fetch 합니다.
이후 workspace 내부에서는 최종 산출물을 `./models/<model-name>.mxq`로 다시 정규화해,
이후 절차가 모두 `./models` 기준으로 동작하도록 맞춥니다.

```bash
python3 examples/run_qb_build.py --model-name resnet50
```

설치된 `mblt_model_zoo.vision` 안에서 표준 fetch에 쓸 수 있는 모델 이름 후보를 보고 싶으면:

```bash
python3 examples/run_qb_build.py --list-model-zoo
```

예를 들어 이름 정규화 기준으로 다음 같은 매칭을 기대합니다.
- `resnet50` -> `mblt_model_zoo.vision.ResNet50`
- `mobilenet_v2` -> `mblt_model_zoo.vision.MobileNetV2`

단, 실제로 materialize 가능한 모델은 설치된 `mblt-model-zoo` 패키지가 제공하는 클래스에 따라 달라집니다.

### 4-b) custom fetching smoke

사전 컴파일된 `.mxq`를 직접 받은 경우에는 로컬 경로를 명시해서 fetch 합니다.

```bash
python3 examples/run_qb_build.py \
  --mxq /path/to/resnet50.mxq \
  --model-name resnet50
```

또는 `models/` 아래에 `resnet50*.mxq`를 두고 `--model-name resnet50`만 써도 됩니다.

### 4-c) custom compile smoke

직접 컴파일하는 경로는 두 가지입니다.

```bash
# (1) ONNX -> .mxq
python3 examples/run_qb_build.py \
  --from-onnx models/yolov7.onnx \
  --use-random-calib \
  --model-name yolov7

# (2) PTH/PT -> ONNX export -> .mxq
#     현재 예제는 resnet50.pth/.pt 기준으로 지원합니다.
python3 examples/run_qb_build.py \
  --from-pth models/resnet50.pth \
  --use-random-calib \
  --model-name resnet50
```

`--from-pth`를 쓰면 기본적으로 `models/<model-name>.onnx`를 중간 산출물로 생성한 뒤,
그 ONNX를 compiler Python API(`qubee`/`qbcompiler`)로 `.mxq` 컴파일합니다.
중간 ONNX 경로를 직접 정하고 싶으면 `--export-onnx-path`를 지정하세요.
단, `--from-pth models/resnet50.pth`는 **torchvision ResNet50 분류 모델 가중치**일 때를 가정합니다.
예를 들어 RetinaFace처럼 backbone 키가 `body.*`로 시작하는 checkpoint는 이 경로로 바로 쓸 수 없고,
해당 아키텍처 전용 ONNX export 또는 사전 컴파일된 `.mxq`/별도 ONNX 준비가 필요합니다.
기본 `target_device`는 `--product`에서 추론합니다.
- `aries` -> `aries-rb`
- `regulus` -> `regulus-rb`

현재 compiler가 요구하는 실제 문자열이 다를 수 있으므로, 필요하면 직접 override 하세요.

```bash
python3 examples/run_qb_build.py \
  --from-onnx models/yolov7.onnx \
  --use-random-calib \
  --model-name yolov7 \
  --target-device aries-rb
```

### 5) .mxq 추론

기본 입력 이미지는 `models/input.jpg`, 클래스 라벨은 `models/labels.txt`를 사용합니다.
이미지가 없으면 synthetic zeros 입력으로 런타임 경로를 검증합니다.
`run_qb_infer.py`는 MXQ가 보고하는 입력 dtype(`Uint8` / `Float32` 등)에 맞춰
이미지 전처리와 synthetic 입력 dtype을 자동으로 맞춥니다.

```bash
python3 examples/run_qb_infer.py \
  --engine-path builds/resnet50.mxq \
  --device 0 \
  --iters 50
```

### 6) 모델 메타 확인

```bash
python3 examples/inspect_qb_model.py builds/resnet50.mxq
```

### 7) 선택: LLM smoke (preview)

Mobilint 문서 기준으로 `qb Runtime`은 v1.2.0부터 **Batch LLM**을 지원하며,
`CacheInfo`, `SequenceBatchParam`, `cache_size` 기반의 low-level sequence inference primitive를 제공합니다.
또한 Mobilint Model Zoo는 transformer / language / multimodal `.mxq`를 Hugging Face group을 통해 제공합니다.

다만 현재 `qb-only`는 vision branch가 기본이며, **LLM custom compile wrapper는 아직 공식 smoke 대상으로 일반화하지 않았습니다.**
이유는 현재 공개 Mobilint 문서 기준으로 **precompiled transformer/LLM `.mxq` fetch**와 **low-level runtime primitive**에 대한 근거는 충분하지만,
**local LLM checkpoint/source model -> qb compiler -> `.mxq`** 경로를 branch public API로 일반화할 만큼 선명한 vendor compile workflow는 아직 부족하기 때문입니다.
따라서 현재 preview 범위는 다음과 같습니다.

- `1) model zoo LLM fetch`
- `2) local precompiled LLM .mxq fetch`
- `4) low-level runtime smoke`
- `5) cache/meta inspect`

위 네 가지를 우선 제공합니다.

현재 완료 기준:
- `prepare_qb_transformer_model.py`로 precompiled LLM `.mxq` 확보
- `run_qb_build.py --mxq ...`로 local fetch 확인
- `run_qb_llm_infer.py`로 single-step / batch low-level runtime smoke 확인
- `inspect_qb_llm_model.py`로 cache/meta inspect 확인

반면 아래 항목은 아직 후속 과제로 남겨둡니다.
- `3) local model/checkpoint -> qb compiler -> LLM .mxq`
- high-level `generate(text)` 스타일 serving helper

위 항목들은 단순 미구현이라기보다, **vendor SDK 공개 지원 범위를 추가로 확인한 뒤 반영할 planned 항목**으로 보는 편이 정확합니다.

참고 문서:
- Batch LLM support added in qb Runtime v1.2.0
- Mobilint Model Zoo provides transformer models as precompiled `.mxq`

```bash
# 7-a) 표준 fetching smoke (Mobilint HF group 의 precompiled transformer/LLM MXQ 확보)
python3 examples/prepare_qb_transformer_model.py \
  --model-id mobilint/Llama-3.2-1B-Instruct

# 위 helper는 snapshot을 models/Llama-3.2-1B-Instruct/ 아래에 받고,
# 최종적으로 ./models/Llama-3.2-1B-Instruct.mxq 로 정규화합니다.

# 7-b) custom fetching smoke (로컬 .mxq 직접 fetch)
python3 examples/run_qb_build.py \
  --mxq models/Llama-3.2-1B-Instruct.mxq \
  --model-name Llama-3.2-1B-Instruct

# 7-c) custom compile smoke (LLM) — planned
# 현재 qb-only는 vision 컴파일 흐름(ONNX / torchvision .pth -> .mxq)을 우선 지원합니다.
# Transformer/LLM custom compile 은 compiler transformer workflow 와
# model-specific export pipeline 정리가 더 필요해 현재는 공식 smoke 에서 제외합니다.
# 특히 현 공개 Mobilint 문서 기준으로는 local LLM source/checkpoint -> qb compiler -> .mxq
# 경로를 branch public API 로 일반화할 만큼 vendor compile workflow 근거가 충분하지 않아
# planned 로만 남겨둡니다.

# 7-d) low-level runtime smoke
# 실제 generate API 가 아니라 Unified SDK sequence runtime
#   infer_sequence(rh, input_array, cache_size=..., batch_params=...)
# 형태로 감싼 cache-aware runtime smoke 입니다.
python3 examples/run_qb_llm_infer.py \
  --engine-path models/Llama-3.2-1B-Instruct.mxq \
  --core-mode global8 \
  --iters 5

# Batch LLM smoke 예시
python3 examples/run_qb_llm_infer.py \
  --engine-path models/Llama-3.2-1B-Instruct.mxq \
  --core-mode global8 \
  --batch-seq-lens 10,80 \
  --iters 3

# 7-e) cache/meta inspect
python3 examples/inspect_qb_llm_model.py models/Llama-3.2-1B-Instruct.mxq --core-mode global8
```

주의 사항:
- 위 LLM smoke는 `generate(text)` 수준의 고수준 serving wrapper가 아니라, 문서에 나온 **cache-aware infer primitive** 기준 smoke 입니다.
- 다만 preview helper도 이제 vendor direct API 대신 Unified SDK sequence runtime API
  `create_sequence_runtime(cfg) -> infer_sequence(rh, input_array, cache_size=..., batch_params=...) -> destroy_sequence_runtime(rh)`
  경로를 우선 검증합니다.
- transformer/LLM MXQ는 여러 코어 모드를 함께 담는 경우가 있어, preview helper는 기본 `core_mode=global8`을 사용합니다.
  `CoreMode::Auto` 오류가 나면 명시적으로 `--core-mode global8` 또는 MXQ가 지원하는 모드를 지정하세요.
- `run_qb_llm_infer.py`는 MXQ가 보고하는 input shape / input dtype에 맞춰 synthetic zeros 입력을 만들어 low-level runtime path를 검증합니다.
- 단일-step smoke에서 MXQ 입력 shape가 `(1, -1, hidden_dim)`처럼 동적 시퀀스 길이를 보고하면,
  preview helper는 `-1` 축을 `1 token`으로 치환해 runtime path만 검증합니다.
- Batch LLM은 `get_cache_infos()`와 `SequenceBatchParam(sequence_length, cache_size, cache_id)`를 쓰는 문서 흐름을 그대로 따릅니다.
- 현재 브랜치의 LLM 완료 기준은 **low-level LLM smoke 통과**입니다.
- `build_unified_LLM(cfg)`는 현재 의도적으로 비워둔 상태입니다. 이는 누락보다는,
  공개 vendor SDK 문서 기준으로 **LLM compile contract를 일반화하기 어렵기 때문**입니다.
- high-level generation/helper 경로는 벤더 경로 의존성과 제약이 남아 있어, 향후 vendor 공식 지원/가이드에 맞춰 업데이트할 예정입니다.

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 컴파일 (.mxq 생성)

```python
from unified_sdk.options import QBBuildOptions
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

# (a) ONNX -> .mxq
cfg = BuildConfig(
    backend="qb",
    model_or_path="models/resnet50.onnx",   # ONNX 경로
    out_dir="builds",
    model_name="resnet50",
    precision="int8",
    input_name="input",
    input_shape=(1, 3, 224, 224),
    calib_data_path=None,                    # 없으면 random calib
    backend_options=QBBuildOptions(
        quantize_method="percentile",
        use_random_calib=True,
        product="aries",
    ),
)
result = build_unified(cfg)
print(result.compiled_model_path)

# (b) 사전 컴파일된 .mxq 확보(fetch): model_or_path 에 .mxq 경로를 그대로 전달
#     cfg = BuildConfig(backend="qb", model_or_path="models/resnet50.mxq", ...)

# (c) PyTorch module 인스턴스를 직접 전달하는 lower-level 경로도 이론상 가능하지만,
#     qb-only 예제/README 기준 권장 custom compile 흐름은
#     .pth/.pt -> ONNX export -> build_unified(cfg) 입니다.
```

### 추론

```python
import numpy as np
from unified_sdk.options import QBVisionRuntimeOptions
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime

cfg = RuntimeConfig(
    backend="qb",
    engine_path="builds/resnet50.mxq",
    input_name="input",
    output_name="output",
    input_shape=(1, 3, 224, 224),
    backend_options=QBVisionRuntimeOptions(device=0, core_mode="auto"),
)
rh = create_runtime(cfg)
y = infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
destroy_runtime(rh)
```

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 Mobilint SDK(`qubee`/`qbruntime`) 위에서 동작하는 통합 추상화 계층이며, 해당 패키지의 라이선스/IP 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 QB(Mobilint ARISE) 어댑터만 노출합니다. 다중 백엔드는 `main` 브랜치에서 사용하세요.
- ARISE 런타임은 **`qbruntime`(QB-RUNTIME)** 을 사용합니다.
- compiler Python API(`qubee` 또는 `qbcompiler`)는 **ONNX**를 입력으로 받아 int8 양자화 `.mxq`를 생성합니다. calibration 데이터셋이
  없으면 `use_random_calib=True`로 smoke 컴파일할 수 있습니다.
- `.mxq`의 입력 layout/dtype은 컴파일 시 결정(compiler `preprocess_dict`)되므로, 추론 입력을 이에 맞춰야 합니다.
- 다중 장치 서버에서는 `MBLT_DEVICE`/`--device`로 장치 ID를 고정하고,
  `MBLT_CORE_MODE`/`--core-mode`는 MXQ 가 실제로 지원하는 모드(`single`, `global4`, `global8`, `auto`)에 맞춰 지정하세요.
- 장치/모델 점검용 CLI: `mobilint-cli status`, `mobilint-cli mxqtool show <mxq>`,
  `mobilint-cli testinfer ...`, `mobilint-cli benchmark ...`.
- `qb Runtime`은 문서상 `Batch LLM`을 지원하며, `cache_size`, `SequenceBatchParam`, `get_cache_infos()` 같은 low-level sequence primitive 를 제공합니다.
- Mobilint Model Zoo는 vision 외에도 transformer / language / multimodal `.mxq`를 제공하지만,
  현재 `qb-only`가 **공식 smoke로 일반화한 custom compile 경로는 vision 우선**입니다.
- 따라서 `qb-only`의 LLM smoke는 현재 **precompiled MXQ fetch + low-level cache-aware infer + cache/meta inspect** 위주로 제공합니다.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_qb_build.py --help`,
  `python3 examples/run_qb_infer.py --help`, `python3 examples/inspect_qb_model.py --help`,
  `python3 examples/prepare_qb_transformer_model.py --help`, `python3 examples/run_qb_llm_infer.py --help`,
  `python3 examples/inspect_qb_llm_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `furiosa-only`, `furiosa-llm-only`)에서 작업하세요.
