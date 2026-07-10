# Unified SDK — Warboy-only (FuriosaAI Warboy)

이 체크아웃(`furiosa-only` 브랜치)은 **FuriosaAI Warboy NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 Warboy 1종으로 좁힌 버전입니다.

`main`의 멀티 백엔드 코드와 동일한 API 표면을 갖되, `rbln-only`·`qb-only`와 동일한 단일-백엔드 패턴을 따릅니다.
컴파일은 **`furiosa-compiler`**(quantized ONNX → `.enf`), 추론은 **`furiosa.runtime`**(sync)을 사용합니다.
(RNGD/LLM 워크로드는 `furiosa-llm-only` 브랜치에서 다룹니다.)

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 Warboy 단일 백엔드 분기입니다.

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
│   ├── run_warboy_build.py         # .enf 확보(fetch) 또는 quantized ONNX→.enf 컴파일(furiosa-compiler)
│   ├── run_warboy_infer.py         # .enf 모델 추론 (furiosa.runtime)
│   └── inspect_warboy_model.py     # .enf 입출력 메타 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (Warboy 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── warboy_build.py         # Warboy 빌드 어댑터 (furiosa-compiler)
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / destroy_runtime
        ├── registry.py
        └── warboy_runtime.py       # Warboy 런타임 어댑터 (furiosa.runtime)
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. `rbln-only`와 동일한 흐름(env → smoke → resnet50
> compile → infer)을 furiosa.quantizer/furiosa-compiler/furiosa.runtime 기준으로 구성했습니다.

---

## 💾 설치 방법

### 1. 저장소 체크아웃

이 브랜치는 두 방식 모두 지원합니다.

- 별도 worktree 폴더 예: `.../furiosa-only/`
- 일반 저장소 루트 예: `.../unified-npu-sdk/`에서 `git switch furiosa-only`

FuriosaAI Warboy 스택은 **공개 APT(`warboy-jammy`) + 공개 pip**로 설치되며, 별도 인증 파일이 필요 없습니다.

### 2. Docker 사전 준비

- `furiosa-only` 검증은 **Docker 기준**으로 진행합니다. 호스트에 `pip install -e .` 같은 로컬 직접 설치는 선택 사항입니다.
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

- **Warboy 커널 드라이버**가 호스트에 설치되어 있어야 합니다 (`furiosactl list`/`info`로 확인).
  자세한 절차는 <https://developer.furiosa.ai/docs/latest/en/> 참조.
- 컨테이너 실행 시 존재하는 장치 노드(`/dev/npu*`)만 `--device`로 전달합니다.
- 기본 컴파일 타깃은 단일 PE 기준 `warboy` 입니다.
- 2 PE 환경에서는 `--target-npu warboy-2pe` 또는 `extra={"target_npu": "warboy-2pe"}` 를 명시해 사용하세요.

### 4. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 `torch`/`torchvision`을 CPU wheel index에서 설치하고, `warboy-jammy` APT suite와
`furiosa-sdk[quantizer]==0.10.2`(+`furiosa-models`)를 이미지에 설치합니다. Furiosa pip 인덱스가
따로 필요하면 `FURIOSA_PIP_INDEX=... ./build.sh` 또는 `./build.sh --furiosa-pip-index <url>`로 지정합니다.

컨테이너 실행 예시:

```bash
docker run -it --security-opt seccomp=unconfined \
  --name furiosa-only \
  --device /dev/npu0:/dev/npu0 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:warboy
```

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
furiosactl list && furiosactl info || true
furiosa-compiler --version || true
python3 -c "import unified_sdk; from furiosa.runtime import sync; print('OK')"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **Warboy 장치가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `furiosa-only`
백엔드를 검증하는 표준 smoke 절차입니다. 추가 wrapper 계층 없이 Unified SDK의 Warboy adapter가
vendor SDK(`furiosa-compiler`/`furiosa.runtime`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
furiosactl list && furiosactl info || true
furiosa-compiler --version || true
python3 -c "import unified_sdk; from furiosa.runtime import sync; print('OK')"

# 4) .enf 확보 또는 컴파일
#    (a) 사전 컴파일된 .enf 를 models/ 에 두었다면 그대로 확보(fetch):
python3 examples/run_warboy_build.py --model-name resnet50
#    (b) quantized ONNX 를 furiosa-compiler 로 컴파일(compile hook, 기본 1 PE):
python3 examples/run_warboy_build.py \
  --from-onnx models/resnet50_quantized.onnx \
  --model-name resnet50

#    (c) 2 PE 환경이라면 target-npu 를 명시:
python3 examples/run_warboy_build.py \
  --from-onnx models/resnet50_quantized.onnx \
  --target-npu warboy-2pe \
  --model-name resnet50

# 5) .enf 추론
#    tests/input.jpg가 없으면 synthetic 입력으로 런타임 경로를 검증합니다.
python3 examples/run_warboy_infer.py \
  --engine-path builds/resnet50.enf \
  --iters 50

# 6) 모델 메타 best-effort 확인
python3 examples/inspect_warboy_model.py builds/resnet50.enf
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 컴파일 (.enf 생성)

```python
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

# (a) quantized ONNX -> .enf (furiosa-compiler)
cfg = BuildConfig(
    backend="warboy",
    model_or_path="models/resnet50_quantized.onnx",  # quantized ONNX 경로
    out_dir="builds",
    model_name="resnet50",
    precision="int8",
    input_name="input",
    input_shape=(1, 3, 224, 224),
    extra={"target_npu": "warboy", "target_ir": "enf"},
)
result = build_unified(cfg)
print(result.compiled_model_path)

# 2 PE 환경 예시:
#     extra={"target_npu": "warboy-2pe", "target_ir": "enf"}

# (b) 사전 컴파일된 .enf 확보(fetch): model_or_path 에 .enf 경로를 그대로 전달
#     cfg = BuildConfig(backend="warboy", model_or_path="models/resnet50.enf", ...)
```

### 추론

```python
import numpy as np
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime

cfg = RuntimeConfig(
    backend="warboy",
    engine_path="builds/resnet50.enf",
    input_name="input",
    output_name="output",
    input_shape=(1, 3, 224, 224),
    extra={"device": None},   # 예: "warboy(0)*2"
)
rh = create_runtime(cfg)
y = infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
destroy_runtime(rh)
```

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 FuriosaAI SDK(`furiosa-compiler`/`furiosa.runtime`/`furiosa.quantizer`) 위에서 동작하는 통합 추상화 계층이며, 해당 패키지의 라이선스/IP 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 Warboy 어댑터만 노출합니다. 다중 백엔드는 `main` 브랜치에서 사용하세요.
- 컴파일러 `furiosa-compiler`는 **quantized ONNX**를 입력으로 받아 `.enf`(int8)를 생성합니다.
  f32 ONNX 는 `furiosa.quantizer`(calibration)로 먼저 양자화해야 합니다 (host validation 참고).
- `.enf`의 입력 dtype/layout은 quantized ONNX 스펙에 따라 고정(보통 int8/uint8)되므로, 추론 입력을 이에 맞춰야 합니다.
- 다중 장치 서버에서는 `FURIOSA_DEVICES`/`--device`(예: `warboy(0)*2`)로 장치를 고정하세요.
- 장치/모델 점검용 CLI: `furiosactl list`, `furiosactl info`, `furiosa-smi info`.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_warboy_build.py --help`,
  `python3 examples/run_warboy_infer.py --help`, `python3 examples/inspect_warboy_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-llm-only`)에서 작업하세요.
