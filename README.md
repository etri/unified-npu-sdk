# Unified SDK — QB-only (Mobilint ARISE)

이 체크아웃(`qb-only` 브랜치)은 **Mobilint ARISE(QB) NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 QB 1종으로 좁힌 버전입니다.

`main`의 멀티 백엔드 코드와 동일한 API 표면을 갖되, `rbln-only`·`trt-only`와 동일한 단일-백엔드 패턴을 따릅니다.
컴파일은 **`qubee`**(ONNX → `.mxq` 양자화 컴파일러), 추론은 **`qbruntime`**(QB-RUNTIME)을 사용합니다. (ARISE는 `maccel`이 아니라 `qbruntime`)

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
├── vendor/                         # (gitignore) Mobilint SDK wheel 배치 위치
│   └── README.md                   #   qubee-*.whl / qbruntime-*.whl
├── examples/
│   ├── run_qb_build.py             # .mxq 확보(fetch) 또는 ONNX→.mxq 컴파일(qubee)
│   ├── run_qb_infer.py             # .mxq 모델 추론 (qbruntime)
│   └── inspect_qb_model.py         # .mxq 요약 정보 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (QB 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── qb_build.py             # QB 빌드 어댑터 (qubee.mxq_compile)
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / destroy_runtime
        ├── registry.py
        └── qb_runtime.py           # QB 런타임 어댑터 (qbruntime)
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. `rbln-only`와 동일한 흐름(env → smoke → resnet50
> compile → infer)을 qubee/qbruntime/mobilint-cli 기준으로 구성했습니다.

---

## 💾 설치 방법

### 1. 저장소 체크아웃 & 벤더 패키지 배치

이 브랜치는 두 방식 모두 지원합니다.

- 별도 worktree 폴더 예: `.../qb-only/`
- 일반 저장소 루트 예: `.../unified-npu-sdk/`에서 `git switch qb-only`

Mobilint SDK(`qubee`, `qbruntime`)는 공개 PyPI에 없으므로, 벤더에게 받은 wheel을 `vendor/`에 둡니다.

```bash
# 예시 1) 별도 worktree
# cd ~/Codings/Micro_DC/qb-only

# Mobilint SDK wheel 배치 (docs.mobilint.com 참조)
cp /path/to/qubee-*.whl     vendor/
cp /path/to/qbruntime-*.whl vendor/
```

### 2. Docker 사전 준비

- `qb-only` 검증은 **Docker 기준**으로 진행합니다. 호스트에 `pip install -e .` 같은 로컬 직접 설치는 선택 사항입니다.
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
  자세한 절차는 <https://docs.mobilint.com/v1.2/en/introduction.html> 참조.
- 컨테이너 실행 시 실제로 존재하는 장치 노드(`/dev/aries*` 또는 `/dev/arise*`)만 `--device`로 전달합니다.
- 코어 모드는 참조 검증 기준 `global8`을 기본값으로 사용하며, `MBLT_CORE_MODE`로 바꿀 수 있습니다.

### 4. 로컬 개발 설치 (선택, 컨테이너 대신 직접)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install onnx
pip install -e .
# Mobilint SDK (vendor-provided)
pip install vendor/qubee-*.whl vendor/qbruntime-*.whl
```

### 5. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 기본적으로 `torch`/`torchvision`을 CPU wheel index
(`https://download.pytorch.org/whl/cpu`)에서 설치하고, `vendor/*.whl`(qubee/qbruntime)을 이미지에
설치합니다. 다른 PyTorch index를 써야 하면 `PYTORCH_INDEX_URL=... ./build.sh` 또는
`./build.sh --pytorch-index-url <url>`로 바꿀 수 있습니다.

컨테이너 실행 예시:

```bash
docker run -it --security-opt seccomp=unconfined \
  --name unified-sdk_qb_dev \
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
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **Mobilint ARISE 장치가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `qb-only`
백엔드를 검증하는 표준 smoke 절차입니다. 추가 wrapper 계층 없이 Unified SDK의 QB adapter가
vendor SDK(`qubee`/`qbruntime`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드 (vendor/ 에 qubee, qbruntime wheel 필요)
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
command -v mobilint-cli && mobilint-cli status || true
python3 -c "import unified_sdk, qbruntime; print('OK')"
python3 -c "import qbruntime; from qbruntime import type as t; print('devices=', t.get_available_device_numbers())"
python3 -c "import qubee; print('qubee=', getattr(qubee, '__version__', 'unknown'))"

# 4) .mxq 확보 또는 컴파일
#    (a) 사전 컴파일된 .mxq 를 models/ 에 두었다면 그대로 확보(fetch):
python3 examples/run_qb_build.py --model-name resnet50
#    (b) ONNX 를 qubee 로 컴파일(compile hook, random calib smoke):
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
- 컴파일러 `qubee`는 **ONNX**를 입력으로 받아 int8 양자화 `.mxq`를 생성합니다. calibration 데이터셋이
  없으면 `use_random_calib=True`로 smoke 컴파일할 수 있습니다.
- `.mxq`의 입력 layout/dtype은 컴파일 시 결정(qubee `preprocess_dict`)되므로, 추론 입력을 이에 맞춰야 합니다.
- 다중 장치 서버에서는 `MBLT_DEVICE`/`--device`로 장치 ID를, `MBLT_CORE_MODE`/`--core-mode`로 코어 모드를 고정하세요.
- 장치/모델 점검용 CLI: `mobilint-cli status`, `mobilint-cli mxqtool show <mxq>`,
  `mobilint-cli testinfer ...`, `mobilint-cli benchmark ...`.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_qb_build.py --help`,
  `python3 examples/run_qb_infer.py --help`, `python3 examples/inspect_qb_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `furiosa-only`, `furiosa-llm-only`)에서 작업하세요.
