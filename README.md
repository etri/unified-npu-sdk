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
# cd ~/users/jskang/uDC/unified-npu-sdk

# Rebellions SDK 사설 인덱스(pypi.rbln.ai) 접근용 자격 파일
mkdir -p .secrets
cat > .secrets/netrc <<'EOF'
machine pypi.rbln.ai
login YOUR_RBLN_USERNAME
password YOUR_RBLN_PASSWORD
EOF
chmod 600 .secrets/netrc
```

### 2. 호스트 사전 요구사항

- **RBLN 드라이버**가 호스트에 설치되어 있어야 합니다 (`rbln-smi`로 확인).
  대부분의 클라우드 서버는 사전 설치되어 있습니다. 자세한 절차는
  <https://docs.rbln.ai/latest/getting_started/installation_guide.html> 참조.
- 컨테이너 실행 시 실제로 존재하는 RBLN 장치만 `--device`로 전달하면 됩니다.
  일부 서버는 `/dev/rbln0`, `/dev/rbln1`만 있고 `/dev/rsdo`는 없을 수 있습니다.
  `./build.sh`는 현재 호스트에서 보이는 `/dev/rbln*`와 선택적 `/dev/rsdo`를 자동 감지해
  `docker run` 예시를 출력합니다.
- NPU가 여러 개인 서버에서는 컴파일/실행 대상을 `RBLN_DEVICES`로 고정하는 것이 안전합니다.
  예: `RBLN_DEVICES=0 python3 examples/run_rbln_build.py`

### 3. 로컬 개발 설치 (선택, 컨테이너 대신 직접)

```bash
# RBLN Portal 계정 필요. ~/.netrc에 pypi.rbln.ai 자격이 있어야 함.
pip install -e .
pip install --extra-index-url https://pypi.rbln.ai/simple rebel-compiler==0.9.4
# host driver 버전에 따라 맞는 버전으로 바꿔야 합니다.
# 예: driver 2.0.1 -> rebel-compiler==0.9.4
```

### 4. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

컨테이너 실행 예시 (현재 호스트에 `/dev/rbln0`, `/dev/rbln1`만 있는 경우):

```bash
docker run -it --security-opt seccomp=unconfined \
  --name unified-sdk_rbln_dev \
  --device /dev/rbln0:/dev/rbln0 \
  --device /dev/rbln1:/dev/rbln1 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  -v /usr/local/bin/rbln-smi:/usr/local/bin/rbln-smi \
  -v /usr/local/bin/rbln-stat:/usr/local/bin/rbln-stat \
  unified-sdk:rbln
```

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
- `Dockerfile` 기본 `rebel-compiler` 버전은 `0.9.4`입니다. 현재 호스트 driver가 다르면 `./build.sh --compiler-version <version>`으로 맞춰 빌드하세요.
- 예제 스크립트는 현재 작업 디렉터리의 checkout root를 우선 사용하므로 `/workspace/unified-sdk`와 `/workspace/unified-npu-sdk` 둘 다 지원합니다.
- 새 백엔드 추가가 필요하면 해당 vendor 브랜치(예: `qb-only`, `furiosa-only`)에서 작업하세요.
