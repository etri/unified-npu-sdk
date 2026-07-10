# Unified SDK — RBLN-only

이 체크아웃(`rbln-only` 브랜치)은 **Rebellions(RBLN) NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 RBLN 1종으로 좁힌 버전입니다.

`main`의 멀티 백엔드(TRT + RBLN) 코드와 동일한 API 표면을 갖되, `trt-only`와 동일한 단일-백엔드 패턴을 따릅니다.

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 RBLN 단일 백엔드 분기입니다.

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
├── .secrets/                       # (gitignore) Rebellions SDK 인증
│   └── netrc                       # 사용자가 직접 생성
├── examples/
│   ├── run_rbln_build.py           # torchvision resnet50 → .rbln 컴파일
│   ├── run_rbln_infer.py           # .rbln 모델 추론
│   └── inspect_rbln_model.py       # .rbln 입출력 메타 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (RBLN 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── rbln_build.py           # RBLN 빌드 어댑터
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / destroy_runtime
        ├── registry.py
        └── rbln_runtime.py         # RBLN 런타임 어댑터
```

---

## 💾 설치 방법

### 1. 저장소 체크아웃 & 인증 파일 생성

이 브랜치는 두 방식 모두 지원합니다.

- 별도 worktree 폴더 예: `.../rbln-only/`
- 일반 저장소 루트 예: `.../unified-npu-sdk/`에서 `git switch rbln-only`

아래 명령은 **현재 체크아웃 루트**에서 실행하면 됩니다.

```bash
# 예시 1) 별도 worktree
# cd ~/Codings/Micro_DC/rbln-only
#
# 예시 2) 일반 저장소 루트에서 rbln-only 브랜치 체크아웃
# cd ~/work/unified-npu-sdk

# Rebellions SDK 사설 인덱스(pypi.rbln.ai) 접근용 자격 파일
mkdir -p .secrets
cat > .secrets/netrc <<'EOF'
machine pypi.rbln.ai
login YOUR_RBLN_USERNAME
password YOUR_RBLN_PASSWORD
EOF
chmod 600 .secrets/netrc
```

### 2. Docker 사전 준비

- `rbln-only` 검증은 **Docker 기준**으로 진행합니다. 호스트에 `pip install -e .` 같은 로컬 직접 설치는 선택 사항입니다.
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

- **RBLN 드라이버**가 호스트에 설치되어 있어야 합니다 (`rbln-smi`로 확인).
  대부분의 클라우드 서버는 사전 설치되어 있습니다. 자세한 절차는
  <https://docs.rbln.ai/latest/getting_started/installation_guide.html> 참조.
- **RBLN Container Toolkit**(`rbln-container-toolkit`)을 설치하는 것을 권장합니다.
  공식 Docker/NPU 연동 경로는 CDI handle(`rebellions.ai/npu=all`) 기반입니다. 자세한 내용은
  <https://docs.rbln.ai/latest/software/system_management/container_toolkit.html> 참조.
- Ubuntu/Debian 예시:

```bash
sudo apt-get update
sudo apt-get install -y rbln-container-toolkit
```

- Container Toolkit 설치 후에는 아래 순서로 CDI/runtime 구성을 완료합니다.

```bash
sudo rbln-ctk cdi generate
sudo rbln-ctk runtime configure
rbln-ctk cdi list
rbln-ctk info
docker run --device rebellions.ai/npu=all -it ubuntu:22.04 rbln-smi
```

- `./build.sh`는 `/var/run/cdi/rbln.yaml`이 있으면 위 CDI 경로를 우선 사용합니다.
- CDI가 없는 호스트에서는 현재 보이는 `/dev/rbln*`와 `rbln-smi`/`rbln-stat` 실행 파일을
  감지해 fallback `docker run` 예시를 출력합니다. 다만 기본 권장 경로는 Container Toolkit + CDI입니다.
- NPU가 여러 개인 서버에서는 컴파일/실행 대상을 `RBLN_DEVICES`로 고정하는 것이 안전합니다.
  예: `RBLN_DEVICES=0 python3 examples/run_rbln_build.py`

### 4. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 기본적으로 `torch`/`torchvision`을 CPU wheel index
(`<https://download.pytorch.org/whl/cpu>`)에서 먼저 설치합니다. RBLN 컴파일 경로에는
CUDA wheel이 필요 없고, CUDA wheel 조합은 compiler frontend 진단을 어렵게 만들 수 있습니다.
다른 PyTorch index를 써야 하면 `PYTORCH_INDEX_URL=... ./build.sh` 또는
`./build.sh --pytorch-index-url <url>`로 바꿀 수 있습니다.

컨테이너 실행 예시 (RBLN Container Toolkit CDI 설정이 완료된 경우):

```bash
docker run -it --security-opt seccomp=unconfined \
  --name unified-sdk_rbln_dev \
  --device rebellions.ai/npu=all \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:rbln
```

CDI가 없는 호스트에서는 `./build.sh`가 감지한 `/dev/rbln*` 기반 fallback 실행 예시를
출력합니다.

컨테이너 내부 점검:

```bash
# repo를 /workspace/unified-sdk 로 직접 마운트한 경우
cd /workspace/unified-sdk

# 부모 디렉터리(uDC)를 /workspace 로 마운트했다면:
# cd /workspace/unified-npu-sdk

rbln-smi
python3 -c "import unified_sdk, rebel; print('OK')"
RBLN_DEVICES=0 python3 examples/run_rbln_build.py
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **RBLN 장치가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `rbln-only`
백엔드를 검증하는 표준 smoke 절차입니다. 추가 wrapper 계층 없이 Unified SDK의
RBLN adapter가 vendor SDK(`rebel`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
command -v rbln-smi && rbln-smi || true
python3 -c "import unified_sdk, rebel; print('OK')"
python3 -c "import rebel; print('npu_is_available=', rebel.npu_is_available())"
python3 -c "import torch, torchvision, rebel; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('rebel=', getattr(rebel, '__version__', 'unknown'))"

# 4) ResNet50 -> .rbln 컴파일
#    기본 smoke는 외부 weight 없이 ResNet50 random-init 모델로 컴파일 경로를 검증합니다.
RBLN_DEVICES=0 python3 examples/run_rbln_build.py \
  --model-name resnet50 \
  --precision fp32 \
  --input-shape 1,3,224,224 \
  --npu "${RBLN_NPU_NAME:-RBLN-CA22}"

# 5) .rbln 추론
#    tests/input.jpg가 없으면 synthetic zeros 입력으로 런타임 경로를 검증합니다.
RBLN_DEVICES=0 python3 examples/run_rbln_infer.py \
  --engine-path builds/resnet50.rbln \
  --device 0 \
  --tensor-type pt \
  --iters 50

# 6) 모델 메타 best-effort 확인
python3 examples/inspect_rbln_model.py builds/resnet50.rbln --device 0
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 컴파일 (.rbln 생성)

```python
import torch
from torchvision.models import resnet50

from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

model = resnet50(weights=None)
model.eval()

cfg = BuildConfig(
    backend="rbln",
    model_or_path=model,
    out_dir="builds",
    model_name="resnet50",
    precision="fp32",
    input_name="input",
    input_shape=(1, 3, 224, 224),
    extra={"npu": "RBLN-CA22"},  # 또는 os.environ["RBLN_NPU_NAME"]
    # bucketing_shapes=[(1, 3, 224, 224), (4, 3, 224, 224)],  # 옵션
)
result = build_unified(cfg)
print(result.compiled_model_path)
```

### 추론

```python
import numpy as np
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime

cfg = RuntimeConfig(
    backend="rbln",
    engine_path="builds/resnet50.rbln",
    input_name="input",
    output_name="output",
    input_shape=(1, 3, 224, 224),
    extra={"tensor_type": "np", "device": 0},
)
rh = create_runtime(cfg)
y = infer(rh, np.random.rand(1, 3, 224, 224).astype(np.float32))
destroy_runtime(rh)
```

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 Rebellions SDK 위에서 동작하는 통합 추상화 계층이며, `rebel-compiler` 패키지의 라이선스/IP 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 RBLN 어댑터만 노출합니다. 다중 백엔드(TRT+RBLN)는 `main` 브랜치에서 사용하세요.
- `types.py`는 RBLN 친화적으로 슬림화되어 있어 `main`의 `BuildConfig`와 일부 필드(`min/opt/max_input_shape`, `use_execute_v3` 등)가 다릅니다. (`input_shape` + 옵션 `bucketing_shapes`로 대체)
- 일부 물리 서버/컨테이너 조합에서는 RBLN 컴파일 시 `BuildConfig.extra["npu"]`로 장치명(예: `RBLN-CA22`)을 명시해야 할 수 있습니다. 예제는 `RBLN_NPU_NAME` 환경 변수를 우선적으로 읽고, 없으면 `RBLN-CA22`를 기본값으로 사용합니다.
- 다중 NPU 서버에서는 `RBLN_DEVICES=0` 또는 `RBLN_DEVICES=1`처럼 장치 ID를 고정해 두는 편이 안전합니다.
- `Dockerfile` 기본 base image는 `ubuntu:22.04`, 기본 `rebel-compiler` 버전은 `0.11.0`입니다. 현재 호스트 driver/SDK 기준이 다르면 `./build.sh --base-image <image> --compiler-version <version>`으로 맞춰 빌드하세요.
- 예제 스크립트는 현재 작업 디렉터리의 checkout root를 우선 사용하므로 `/workspace/unified-sdk`와 `/workspace/unified-npu-sdk` 둘 다 지원합니다.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_rbln_build.py --help`,
  `python3 examples/run_rbln_infer.py --help`, `python3 examples/inspect_rbln_model.py --help`로 확인하세요.
- 새 백엔드 추가가 필요하면 해당 vendor 브랜치(예: `qb-only`, `furiosa-only`)에서 작업하세요.
