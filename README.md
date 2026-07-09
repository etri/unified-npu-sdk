# Unified SDK — RNGD-only (FuriosaAI RNGD / furiosa-llm)

이 체크아웃(`furiosa-llm-only` 브랜치)은 **FuriosaAI RNGD NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 RNGD 1종으로 좁힌 버전입니다.

`rbln-only`·`qb-only`·`furiosa-only`와 동일한 단일-백엔드 골격을 따르되, **RNGD는 LLM 스택**이라
빌드/추론의 의미가 다릅니다. 모델 준비는 **`furiosa-llm`의 `ArtifactBuilder`**, 서빙은 **`furiosa_llm.LLM`**을 사용하며,
`runtime.infer`는 numpy 추론이 아니라 **LLM 텍스트 생성(프롬프트 → 텍스트)**입니다(`generate` 별칭 제공).
(vision 워크로드인 Warboy는 `furiosa-only` 브랜치에서 다룹니다.)

---

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**의
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물의 RNGD(LLM) 단일 백엔드 분기입니다.

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
│   ├── run_rngd_build.py           # HF 아티팩트/모델 확보(fetch) 또는 ArtifactBuilder AOT 컴파일
│   ├── run_rngd_infer.py           # LLM 텍스트 생성 (프롬프트 → 텍스트)
│   └── inspect_rngd_model.py       # 아티팩트/모델 메타 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (LLM 친화적으로 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── rngd_build.py           # RNGD 빌드 어댑터 (furiosa-llm ArtifactBuilder)
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / generate / destroy_runtime
        ├── registry.py
        └── rngd_runtime.py         # RNGD 런타임 어댑터 (furiosa_llm.LLM.generate)
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. 다른 워크트리와 동일한 골격이되, RNGD는 LLM 흐름
> (env → generate smoke → AOT 아티팩트 빌드 → 아티팩트 generate)으로 구성했습니다.

---

## 💾 설치 방법

### 1. 저장소 체크아웃

이 브랜치는 두 방식 모두 지원합니다.

- 별도 worktree 폴더 예: `.../furiosa-llm-only/`
- 일반 저장소 루트 예: `.../unified-npu-sdk/`에서 `git switch furiosa-llm-only`

FuriosaAI RNGD 스택은 **공개 APT(OS codename suite) + 공개 pip(`furiosa-llm`)**로 설치되며, 별도 인증 파일이 필요 없습니다.
(Warboy의 `warboy-jammy` suite와 다릅니다.)

### 2. Docker 사전 준비

- `furiosa-llm-only` 검증은 **Docker 기준**으로 진행합니다. 호스트에 `pip install -e .` 같은 로컬 직접 설치는 선택 사항입니다.
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

- **RNGD 커널 드라이버**가 호스트에 설치되어 있어야 합니다 (`furiosa-smi info`로 확인).
  자세한 절차는 <https://developer.furiosa.ai/latest/en/> 참조.
- 컨테이너 실행 시 존재하는 장치 노드(`/dev/rngd*`)만 `--device`로 전달합니다.

### 4. 로컬 개발 설치 (선택, 컨테이너 대신 직접)

```bash
pip install -e .
# FuriosaAI RNGD LLM 스택 (torch/transformers 는 furiosa-llm 의존성으로 설치됨)
pip install furiosa-llm
# 시스템: APT OS codename suite (furiosa-driver-rngd, furiosa-smi)
```

### 5. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 `warboy-jammy`가 아닌 **OS codename suite**로 furiosa APT 저장소를 추가하고
`furiosa-smi`를 설치하며, `furiosa-llm`을 pip로 설치합니다. Furiosa pip 인덱스가 따로 필요하면
`FURIOSA_PIP_INDEX=... ./build.sh` 또는 `./build.sh --furiosa-pip-index <url>`로 지정합니다.

컨테이너 실행 예시:

```bash
docker run -it --security-opt seccomp=unconfined \
  --name unified-sdk_rngd_dev \
  --device /dev/rngd0:/dev/rngd0 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:rngd
```

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
furiosa-smi info || true
python3 -c "import unified_sdk; from furiosa_llm import LLM, SamplingParams; print('OK')"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **RNGD 장치가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `furiosa-llm-only`
백엔드를 검증하는 표준 smoke 절차입니다. 추가 wrapper 계층 없이 Unified SDK의 RNGD adapter가
vendor SDK(`furiosa-llm`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드
./build.sh

# 2) build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
furiosa-smi info || true
python3 -c "import unified_sdk; from furiosa_llm import LLM, SamplingParams; print('OK')"

# 4) 모델 확보 또는 컴파일
#    (a) 사전 빌드된 HF 아티팩트/모델 id 확보(fetch, 기본):
python3 examples/run_rngd_build.py --model furiosa-ai/Qwen2.5-0.5B-Instruct
#    (b) ArtifactBuilder 로 AOT 컴파일(compile hook, 무거움):
python3 examples/run_rngd_build.py \
  --model furiosa-ai/Qwen2.5-0.5B-Instruct \
  --compile --tensor-parallel-size 1

# 5) LLM 텍스트 생성
python3 examples/run_rngd_infer.py \
  --engine-path furiosa-ai/Qwen2.5-0.5B-Instruct \
  --prompt "What is the capital of France?" \
  --chat

# 6) 아티팩트/모델 메타 확인
python3 examples/inspect_rngd_model.py furiosa-ai/Qwen2.5-0.5B-Instruct
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 모델 준비 (아티팩트)

```python
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

# (a) fetch (기본): HF 모델 id 또는 기존 아티팩트 dir 를 그대로 사용
cfg = BuildConfig(backend="rngd", model_or_path="furiosa-ai/Qwen2.5-0.5B-Instruct")
result = build_unified(cfg)
print(result.compiled_model_path)   # 모델 id 또는 아티팩트 dir

# (b) AOT 컴파일 (ArtifactBuilder): extra={"compile": True}
cfg = BuildConfig(
    backend="rngd",
    model_or_path="furiosa-ai/Qwen2.5-0.5B-Instruct",
    out_dir="artifacts",
    model_name="qwen2_5_0_5b",
    tensor_parallel_size=1,
    extra={"compile": True},
)
result = build_unified(cfg)
print(result.compiled_model_path)   # 아티팩트 디렉터리
```

### 텍스트 생성

```python
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, generate, destroy_runtime

cfg = RuntimeConfig(
    backend="rngd",
    engine_path="furiosa-ai/Qwen2.5-0.5B-Instruct",  # 아티팩트 dir 또는 모델 id
    max_tokens=128,
    temperature=0.7,
    top_p=0.3,
    top_k=100,
)
rh = create_runtime(cfg)
text = generate(rh, "What is the capital of France?")   # infer 의 LLM 별칭
print(text)
destroy_runtime(rh)
```

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 FuriosaAI SDK(`furiosa-llm`) 위에서 동작하는 통합 추상화 계층이며, 해당 패키지의 라이선스/IP 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 RNGD(LLM) 어댑터만 노출합니다. 다중 백엔드는 `main` 브랜치에서 사용하세요.
- RNGD는 **LLM 스택**이라 `runtime.infer`가 텍스트 생성(프롬프트 → 텍스트)입니다. numpy 추론이 아닙니다.
  가독성을 위해 `generate`를 `infer`의 별칭으로 제공합니다.
- 모델 준비 어댑터는 두 경로를 지원합니다: **fetch(기본)** = HF 모델 id/기존 아티팩트를 그대로 사용,
  **compile 훅** = `ArtifactBuilder`로 AOT 컴파일(무거움). `run_rngd_build.py --compile`.
- chat 모델은 `tokenizer.apply_chat_template`로 프롬프트를 감싸는 것이 정석입니다(`run_rngd_infer.py --chat`).
- 다중 장치/병렬은 `--tensor-parallel-size`(빌드) 및 `--devices`(런타임, 예: `npu:0`)로 지정합니다.
- 장치/상태 점검용 CLI: `furiosa-smi info`, `furiosa-smi info --full`.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_rngd_build.py --help`,
  `python3 examples/run_rngd_infer.py --help`, `python3 examples/inspect_rngd_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-only`)에서 작업하세요.
