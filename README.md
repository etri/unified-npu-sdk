# Unified SDK — RNGD-only (FuriosaAI RNGD / furiosa-llm)

이 체크아웃(`furiosa-llm-only` 브랜치)은 **FuriosaAI RNGD NPU 전용**으로 단일 백엔드만 노출합니다.
공통 추상화(`build/`, `runtime/`)는 그대로 유지하면서, 어댑터·예제·컨테이너 구성을 RNGD 1종으로 좁힌 버전입니다.

`rbln-only`·`qb-only`·`furiosa-only`와 동일한 단일-백엔드 골격을 따르되, **RNGD는 LLM 스택**이라
빌드/추론의 의미가 다릅니다. **공식 smoke 기준점은 `furiosa_llm.LLM` 기반 fetch + generate** 이며,
custom model 검증은 **`fxb build` + `LLM(..., fxb=...)`** 경로로 연결합니다. 서빙은 **`furiosa_llm.LLM`**을 사용하며,
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
│   ├── prepare_rngd_local_model.py  # custom smoke 용 HF snapshot/local model 준비
│   ├── run_rngd_build.py           # HF 모델 id/local path 전달(fetch) 또는 FXB 빌드
│   ├── run_rngd_infer.py           # LLM 텍스트 생성 (프롬프트 → 텍스트)
│   └── inspect_rngd_model.py       # 모델/FXB 메타 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # 공통 데이터 구조 (LLM 친화적으로 슬림화)
    ├── build/
    │   ├── __init__.py
    │   ├── api.py                  # build_unified
    │   ├── registry.py
    │   └── rngd_build.py           # RNGD 빌드 어댑터 (fetch 기본, 선택적 FXB build)
    └── runtime/
        ├── __init__.py
        ├── api.py                  # create_runtime / infer / generate / destroy_runtime
        ├── registry.py
        └── rngd_runtime.py         # RNGD 런타임 어댑터 (furiosa_llm.LLM / LLM(..., fxb=...))
```

> `builds/host_validation_tools/`는 벤더 에스컬레이션용 로컬 재현 팩입니다. `builds/`는 gitignore
> 대상이라 저장소에는 포함되지 않습니다. 다른 워크트리와 동일한 골격이되, RNGD는 LLM 흐름
> (env → model id generate smoke → local path FXB build smoke → explicit FXB generate)으로 구성했습니다.

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

- **RNGD 커널 드라이버**와 `furiosa-smi`가 호스트에 설치되어 있어야 합니다.
  `furiosa-smi info`가 정상 출력되면 이 사전 요구사항은 충족된 상태입니다.
  자세한 절차는 Furiosa 공식 문서의 Get Started / Device Management 경로
  (<https://developer.furiosa.ai/latest/en/>)를 참조하세요.
- 컨테이너 실행 시 존재하는 장치 노드만 `--device`로 전달합니다.
  호스트 레이아웃은 드라이버 버전에 따라 `/dev/rngd*` 또는 `/dev/rngd/`
  아래 계층형 문자 장치로 보일 수 있습니다.

### 4. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 `warboy-jammy`가 아닌 **OS codename suite**로 furiosa APT 저장소를 추가하고
`furiosa-smi`를 설치하며, `furiosa-llm`을 pip로 설치합니다. Furiosa pip 인덱스가 따로 필요하면
`FURIOSA_PIP_INDEX=... ./build.sh` 또는 `./build.sh --furiosa-pip-index <url>`로 지정합니다.
또한 런타임 장치는 `/dev/rngd*`, `/dev/rngd/` 레이아웃을 자동 감지합니다.

컨테이너 실행 예시:

```bash
docker run -it --security-opt seccomp=unconfined \
  --name furiosa-llm-only \
  --device /dev/rngd0:/dev/rngd0 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:rngd
```

장치가 `/dev/rngd/` 아래에 보이는 호스트라면 `./build.sh`가 해당 문자 장치를 자동으로 여러 개
`--device`로 나열합니다. 필요하면 `./build.sh --device /dev/rngd`처럼 디렉터리를 직접 지정할 수도 있습니다.

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
furiosa-smi info || true
python3 -c "import unified_sdk; from furiosa_llm import LLM, SamplingParams; print('OK')"
```

custom FXB build prerequisite:

- `fxb build` custom smoke 는 컨테이너 내부에서 추가 빌드 툴체인을 필요로 할 수 있습니다.
- 현재 Dockerfile 은 이를 위해 `build-essential`, `python3-dev` 를 함께 설치합니다.
- 따라서 이 문서의 custom smoke 를 처음 시도하거나 Dockerfile 변경 후 다시 시도할 때는 `./build.sh`로 이미지를 다시 빌드해야 합니다.

custom local model 준비:

- `models/`와 `artifacts/`는 `.gitignore` 대상이라 저장소 clone/pull 만으로는 준비되지 않습니다.
- custom smoke 는 지원 모델의 **upstream/raw Hugging Face snapshot/local copy** 를 repo 내부 `models/` 아래에 미리 받아두는 전제를 둡니다.
- 현재 `./build.sh`가 마운트하는 `/workspace/unified-sdk` 아래 경로를 그대로 쓰면 추가 `-v /host/models:/models` 없이 진행할 수 있습니다.
- Furiosa 공식 FXB 문서에서 명시적으로 build 예시로 드는 모델은 `Qwen/Qwen3-8B-FP8` 입니다. 현재 custom smoke 예시도 그 기준을 따릅니다.

호스트 예시:

```bash
cd ~/unified-npu-sdk/furiosa-llm-only
mkdir -p models

# huggingface_hub 가 없으면 먼저 설치
python3 -m pip install --user -U huggingface_hub

# custom FXB smoke 용 upstream/raw model snapshot 준비
python3 examples/prepare_rngd_local_model.py \
  --model Qwen/Qwen3-8B-FP8
```

위 명령이 끝나면 기본적으로 아래 경로가 준비됩니다.

```bash
models/Qwen3-8B-FP8
```

직접 `huggingface_hub` CLI를 쓰고 싶다면 동일하게 다음도 가능합니다.

```bash
hf download Qwen/Qwen3-8B-FP8 \
  --local-dir ./models/Qwen3-8B-FP8
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
fxb --help || true

# 4-A) 표준 smoke: model id -> generate
python3 examples/run_rngd_build.py --model furiosa-ai/Qwen2.5-0.5B-Instruct
python3 examples/run_rngd_infer.py \
  --engine-path furiosa-ai/Qwen2.5-0.5B-Instruct \
  --prompt "What is the capital of France?" \
  --chat
python3 examples/inspect_rngd_model.py furiosa-ai/Qwen2.5-0.5B-Instruct

# 4-B) custom smoke: local model path -> fxb build -> LLM(..., fxb=...) -> generate
#      예시 local path 는 supported architecture 의 upstream/raw HF snapshot/local copy 여야 합니다.
#      `furiosa-ai/...` prebuilt artifact repo 는 이 build 입력으로 쓰지 않습니다.
#      이 경로는 Dockerfile 의 build toolchain 변경이 반영된 이미지를 전제로 합니다.
python3 examples/prepare_rngd_local_model.py \
  --model Qwen/Qwen3-8B-FP8
python3 examples/run_rngd_build.py \
  --model models/Qwen3-8B-FP8 \
  --fxb-build \
  --model-name qwen3_8b_fp8 \
  --tensor-parallel-size 1
python3 examples/run_rngd_infer.py \
  --engine-path models/Qwen3-8B-FP8 \
  --fxb-path artifacts/qwen3_8b_fp8.fxb \
  --prompt "What is the capital of France?" \
  --chat
python3 examples/inspect_rngd_model.py models/Qwen3-8B-FP8 \
  --fxb-path artifacts/qwen3_8b_fp8.fxb
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### 모델 준비

```python
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

# (a) fetch (기본): HF 모델 id 또는 로컬 모델 경로를 그대로 사용
cfg = BuildConfig(backend="rngd", model_or_path="furiosa-ai/Qwen2.5-0.5B-Instruct")
result = build_unified(cfg)
print(result.compiled_model_path)   # 모델 id 또는 로컬 모델 경로

# (b) custom smoke: FXB 빌드
cfg = BuildConfig(
    backend="rngd",
    model_or_path="models/Qwen3-8B-FP8",
    out_dir="artifacts",
    model_name="qwen3_8b_fp8",
    tensor_parallel_size=1,
    extra={"build_mode": "fxb_build"},
)
result = build_unified(cfg)
print(result.compiled_model_path)   # artifacts/qwen3_8b_fp8.fxb
```

### 텍스트 생성

```python
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, generate, destroy_runtime

cfg = RuntimeConfig(
    backend="rngd",
    engine_path="furiosa-ai/Qwen2.5-0.5B-Instruct",  # 모델 id 또는 로컬 모델 경로
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

```python
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, generate, destroy_runtime

cfg = RuntimeConfig(
    backend="rngd",
    engine_path="models/Qwen3-8B-FP8",
    fxb_path="artifacts/qwen3_8b_fp8.fxb",
    max_tokens=128,
)
rh = create_runtime(cfg)
text = generate(rh, "What is the capital of France?")
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
- 표준 smoke 는 **model id -> generate** 입니다. 공식 quick start 도 이 경로를 기준으로 합니다.
- custom smoke 는 **local model path -> fxb build -> LLM(..., fxb=...) -> generate** 입니다.
- custom smoke 에서는 `furiosa-ai/...` prebuilt artifact repo 대신 upstream/raw model snapshot 을 사용해야 합니다.
- 현재 custom smoke 예시는 Furiosa 공식 FXB 문서의 build 예시인 `Qwen/Qwen3-8B-FP8` 를 따릅니다.
- custom local model 은 repo 내부 `models/` 같은 gitignored 경로에 별도 준비해야 합니다. 저장소에는 포함되지 않습니다.
- `examples/prepare_rngd_local_model.py`는 custom smoke 전용 local snapshot 준비 예제입니다.
- `FXB`는 Furiosa-LLM의 권장 compiled-artifact 형식입니다. `fxb build`, `fxb check`, `fxb add`는
  `furiosa-llm` 패키지와 함께 제공됩니다.
- chat 모델은 `tokenizer.apply_chat_template`로 프롬프트를 감싸는 것이 정석입니다(`run_rngd_infer.py --chat`).
- 다중 장치/병렬은 `--tensor-parallel-size`(빌드) 및 `--devices`(런타임, 예: `npu:0`)로 지정합니다.
- 장치/상태 점검용 CLI: `furiosa-smi info`, `furiosa-smi info --full`.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_rngd_build.py --help`,
  `python3 examples/run_rngd_infer.py --help`, `python3 examples/inspect_rngd_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-only`)에서 작업하세요.
