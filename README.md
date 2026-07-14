# Unified SDK — QB-only (Mobilint ARISE)

이 체크아웃(`qb-only` 브랜치)은 **Mobilint ARISE(QB) NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 QB 1종으로 좁힌 버전입니다.

`main`의 멀티 백엔드 코드와 동일한 API 표면을 갖되, `rbln-only`·`trt-only`와 동일한 단일-백엔드 패턴을 따릅니다.
컴파일은 **Mobilint compiler Python API**(`qubee` 또는 `qbcompiler`로 노출될 수 있음), 추론은 **`qbruntime`**(QB-RUNTIME)을 사용합니다. `qb-only`는 Mobilint 공식 문서 흐름에 맞춰 **공식 `qbcompiler` Docker 이미지 + 벤더 제공 `qbcompiler` wheel + `pip install mobilint-qb-runtime` + `apt install mobilint-cli`** 조합을 기본 경로로 사용합니다. (ARISE는 `maccel`이 아니라 `qbruntime`)

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 QB(Mobilint ARISE) 단일 백엔드 분기입니다.

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
├── vendor/                         # (gitignore) Mobilint compiler wheel 배치 위치
│   └── README.md                   #   qbcompiler-*.whl
├── examples/
│   ├── run_qb_build.py             # .mxq 확보(fetch) 또는 ONNX→.mxq 컴파일(compiler Python API)
│   ├── run_qb_infer.py             # .mxq 모델 추론 (qbruntime)
│   └── inspect_qb_model.py         # .mxq 요약 정보 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (QB 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── qb_build.py             # QB 빌드 어댑터 (qubee/qbcompiler mxq_compile)
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / destroy_runtime
        ├── registry.py
        └── qb_runtime.py           # QB 런타임 어댑터 (qbruntime)
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. `rbln-only`와 동일한 흐름(env → smoke → resnet50
> compile → infer)을 compiler Python API(qubee/qbcompiler) / qbruntime / mobilint-cli 기준으로 구성했습니다.

---

## 💾 설치 방법

### 1. 저장소 체크아웃 & 컴파일러 wheel 배치

이 브랜치는 두 방식 모두 지원합니다.

- 별도 worktree 폴더 예: `.../qb-only/`
- 일반 저장소 루트 예: `.../unified-npu-sdk/`에서 `git switch qb-only`

Mobilint 공식 문서 기준으로 SDK qb는 `Driver / qb Runtime / qb Compiler`로 나뉩니다. 이 브랜치는:

- **Compiler base**: Mobilint 공식 `qbcompiler` Docker 이미지
- **Compiler Python API**: 벤더 제공 `qbcompiler-*.whl` (`qubee` 또는 `qbcompiler` import로 노출될 수 있음)
- **Runtime**: `pip install mobilint-qb-runtime`
- **CLI Utility**: `apt install mobilint-cli`

조합을 기본 경로로 사용합니다. 따라서 `vendor/`에는 **`qbcompiler` compiler wheel만** 둡니다. 패키지 버전에 따라 compiler Python import 이름은 `qubee` 또는 `qbcompiler`일 수 있습니다.

> 권장: `vendor/`에는 `qbcompiler` wheel을 **한 버전만** 두세요. 여러 버전을 같이 두면 어떤 wheel 기준으로
> base image를 추론할지 헷갈릴 수 있으므로, 테스트에 사용할 버전 하나만 남기는 것이 안전합니다.

```bash
# 예시 1) 별도 worktree
# cd ~/Codings/Micro_DC/qb-only

# Mobilint compiler wheel 배치 (docs.mobilint.com 참조)
cp /path/to/qbcompiler-*.whl vendor/
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

문제 해결 힌트:

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
- 코어 모드는 참조 검증 기준 `global8`을 기본값으로 사용하며, `MBLT_CORE_MODE`로 바꿀 수 있습니다.

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
python3 -c "import importlib, pkgutil; m = next((importlib.import_module(n) for n in ('qubee', 'qbcompiler') if pkgutil.find_loader(n)), None); print('compiler_pkg=', getattr(m, '__name__', 'missing'))"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **Mobilint ARISE 장치가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `qb-only`
백엔드를 검증하는 표준 smoke 절차입니다. 추가 wrapper 계층 없이 Unified SDK의 QB adapter가
vendor SDK(compiler Python API `qubee`/`qbcompiler`, runtime `qbruntime`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드 (vendor/ 에 qbcompiler wheel 필요, runtime 은 pip 설치)
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
command -v mobilint-cli && mobilint-cli status || true
python3 -c "import unified_sdk, qbruntime; print('OK')"
python3 -c "import qbruntime; from qbruntime import type as t; print('devices=', t.get_available_device_numbers())"
python3 -c "import importlib, pkgutil; m = next((importlib.import_module(n) for n in ('qubee', 'qbcompiler') if pkgutil.find_loader(n)), None); print('compiler_pkg=', getattr(m, '__name__', 'missing'), 'version=', getattr(m, '__version__', 'unknown') if m else 'n/a')"

# 4) .mxq 확보 또는 컴파일
#    (a) 사전 컴파일된 .mxq 를 models/ 에 두었거나,
#        Mobilint 공식 문서 흐름대로 ~/.mblt_model_zoo/... 에 이미 생성돼 있다면 그대로 확보(fetch):
python3 examples/run_qb_build.py --model-name resnet50
#    (b) ONNX 를 compiler Python API(qubee/qbcompiler)로 컴파일(compile hook, random calib smoke):
python3 examples/run_qb_build.py \
  --from-onnx models/resnet50.onnx \
  --use-random-calib \
  --model-name resnet50

# 5) .mxq 추론
#    tests/input.jpg가 없으면 synthetic zeros 입력으로 런타임 경로를 검증합니다.
python3 examples/run_qb_infer.py \
  --engine-path builds/resnet50.mxq \
  --device 0 \
  --core-mode global8 \
  --iters 50

# 6) 모델 메타 확인
python3 examples/inspect_qb_model.py builds/resnet50.mxq
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 컴파일 (.mxq 생성)

```python
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

# (a) ONNX -> .mxq (qubee compile hook)
cfg = BuildConfig(
    backend="qb",
    model_or_path="models/resnet50.onnx",   # ONNX 경로
    out_dir="builds",
    model_name="resnet50",
    precision="int8",
    input_name="input",
    input_shape=(1, 3, 224, 224),
    calib_data_path=None,                    # 없으면 random calib
    extra={"quantize_method": "percentile", "use_random_calib": True, "core_mode": "global8"},
)
result = build_unified(cfg)
print(result.compiled_model_path)

# (b) 사전 컴파일된 .mxq 확보(fetch): model_or_path 에 .mxq 경로를 그대로 전달
#     cfg = BuildConfig(backend="qb", model_or_path="models/resnet50.mxq", ...)
```

### 추론

```python
import numpy as np
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime

cfg = RuntimeConfig(
    backend="qb",
    engine_path="builds/resnet50.mxq",
    input_name="input",
    output_name="output",
    input_shape=(1, 3, 224, 224),
    extra={"device": 0, "core_mode": "global8"},
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
- ARISE 런타임은 **`qbruntime`(QB-RUNTIME)** 을 사용합니다. 구형 ARIES용 `maccel`이 아닙니다.
- compiler Python API(`qubee` 또는 `qbcompiler`)는 **ONNX**를 입력으로 받아 int8 양자화 `.mxq`를 생성합니다. calibration 데이터셋이
  없으면 `use_random_calib=True`로 smoke 컴파일할 수 있습니다.
- `.mxq`의 입력 layout/dtype은 컴파일 시 결정(compiler `preprocess_dict`)되므로, 추론 입력을 이에 맞춰야 합니다.
- 다중 장치 서버에서는 `MBLT_DEVICE`/`--device`로 장치 ID를, `MBLT_CORE_MODE`/`--core-mode`로 코어 모드를 고정하세요.
- 장치/모델 점검용 CLI: `mobilint-cli status`, `mobilint-cli mxqtool show <mxq>`,
  `mobilint-cli testinfer ...`, `mobilint-cli benchmark ...`.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_qb_build.py --help`,
  `python3 examples/run_qb_infer.py --help`, `python3 examples/inspect_qb_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `furiosa-only`, `furiosa-llm-only`)에서 작업하세요.
