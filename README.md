# Unified SDK

Unified SDK는 여러 AI 가속기 백엔드를 하나의 공통 개발 흐름으로 다루기 위한 통합 SDK입니다.  
이 저장소의 `main` 브랜치는 각 vendor branch에서 검증한 결과를 바탕으로, 공통 API와 공통 실행 흐름을 다시 정리해가는 **멀티밴더 통합 브랜치**입니다.

## 정의

Unified SDK는 특정 벤더 SDK를 대체하는 새 compiler/runtime가 아니라,  
각 벤더가 제공하는 compiler, runtime, model artifact, serving stack을 **공통 API와 공통 작업 순서**로 감싸는 상위 통합 계층입니다.

즉 이 프로젝트가 제공하려는 것은 다음과 같습니다.

- backend마다 제각각인 build / runtime / infer 흐름의 공통화
- vendor별 예제와 smoke test의 정리
- 하나의 저장소 안에서 여러 NPU 백엔드를 비교하고 검증할 수 있는 작업 기반

현재 `main`은 “모든 backend가 완전히 동일하게 동작하는 최종 제품”이라기보다,
**공통 골격을 먼저 맞추고 backend별 차이를 명시적으로 관리하는 통합 작업 공간**으로 보는 것이 가장 정확합니다.

## 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내  
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**에서 수행한  
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물입니다.

이 SDK는 TensorRT, Rebellions, Furiosa, Mobilint 등 서로 다른 가속기 환경을 대상으로,
모델 빌드와 런타임 생성, 그리고 예제 추론 흐름을 최대한 비슷한 방식으로 다루는 것을 목표로 합니다.

### 현재 통합 상태

- `main`은 branch별 history import를 완료한 상태입니다.
- 공통 Unified SDK public API 골격과 `./build.sh --backend ...` dispatcher가 정리되어 있습니다.
- backend별 지원 수준은 아직 서로 다르며, `planned`, `unsupported`, `known issue`를 그대로 유지합니다.
- 따라서 `main`은 **통합 방향과 실행 진입점이 정리된 브랜치**로 이해하는 것이 적절합니다.

### 현재 포함된 backend

| Backend | Track | Docker | 현재 상태 |
| --- | --- | --- | --- |
| `qb` | vision + low-level LLM runtime | unified | vision 동작, LLM runtime preview, LLM build planned |
| `rbln` | vision + LLM | unified | API 구현 완료, container compile known issue 메모 유지 |
| `warboy` | vision | unified | build / infer / inspect 흐름 구현 |
| `rngd` | LLM | unified | generate 경로 구현, 일부 `fxb build`는 vendor toolchain 이슈 이력 있음 |
| `trt` | vision / LLM | split flavor | `vision`, `llm` 분리 Docker, LLM compile 일부 unsupported |

## 주요 기능

| 구분 | 설명 |
| --- | --- |
| 모델 빌드 | backend별 compiler/toolchain을 공통 entrypoint에서 호출 |
| 런타임 생성 | compiled artifact 또는 vendor runtime reference를 공통 API로 로드 |
| 예제 추론 | backend별 build / infer / generate smoke entry 제공 |
| 멀티백엔드 정리 | vendor별 Docker, example, known issue를 한 저장소에서 관리 |
| API 통일 | vision / LLM public API 이름을 가능한 한 통일 |

### 공통 public API

Vision:

- `build_unified(cfg)`
- `create_runtime(cfg)`
- `infer(...)`
- `destroy_runtime(rh)`

LLM:

- `build_unified_LLM(cfg)`
- `create_runtime_LLM(cfg)`
- `infer_LLM(...)`
- `generate_LLM(...)`
- `destroy_runtime_LLM(rh)`

메모:

- API 이름은 공통이지만 backend별 semantics가 완전히 동일한 것은 아닙니다.
- 예를 들어 `qb` LLM은 high-level text generation보다 low-level cache-aware infer 성격이 강합니다.

## 프로젝트 구조

```text
main/
├── README.md
├── build.sh                    # backend dispatcher
├── Dockers/                    # backend/flavor별 Dockerfiles + requirements
├── scripts/                    # backend별 build launcher
├── examples/                   # vendor/track별 smoke examples
├── src/unified_sdk/
│   ├── __init__.py
│   ├── types.py                # 공통 config / handle 정의
│   ├── build/                  # build adapters + registry
│   └── runtime/                # runtime adapters + registry
├── docs/                       # 설계/검토 자료
└── vendor/                     # vendor 관련 참고 자료
```

이 구조의 핵심은 다음과 같습니다.

- `build.sh`는 사용자가 가장 먼저 만나는 공통 진입점입니다.
- 실제 Docker build 로직은 `scripts/build_<backend>.sh`에 분리되어 있습니다.
- backend별 Dockerfile과 requirements는 `Dockers/` 아래에 정리되어 있습니다.
- 실제 검증용 흐름은 `examples/` 기준으로 따라갈 수 있습니다.

## 설치방법

### 1. 저장소 준비

일반적인 경우:

```bash
git clone https://github.com/etri/unified-npu-sdk.git
cd unified-npu-sdk/main
```

이미 vendor별 checkout이 따로 있는 경우에는,
상위 디렉터리 아래에 `main`만 별도 폴더로 받는 방식이 더 안전합니다.

```bash
cd ~/unified-npu-sdk
git clone --branch main --single-branch https://github.com/etri/unified-npu-sdk.git main
cd main
```

메모:

- 위 명령은 `~/unified-npu-sdk/main` 폴더를 새로 생성합니다.
- `~/unified-npu-sdk/main` 폴더가 이미 있으면 clone이 실패할 수 있으니, 그 경우에는 폴더명을 바꾸거나 비운 뒤 다시 시도합니다.

### 2. backend별 Docker 이미지 빌드

```bash
./build.sh --backend qb
./build.sh --backend rbln
./build.sh --backend warboy
./build.sh --backend rngd
./build.sh --backend trt --flavor vision
./build.sh --backend trt --flavor llm
```

각 backend script는 빌드가 끝나면 해당 환경에 맞는 `docker run ...` 예시를 직접 출력합니다.

### 3. main 전용 이미지/컨테이너 이름

`main` 브랜치에서는 vendor branch와 충돌하지 않도록 이미지/컨테이너 이름도 별도로 사용합니다.

| Backend | 기본 이미지 태그 | 기본 컨테이너 이름 |
| --- | --- | --- |
| `qb` | `unified-sdk:main-qb` | `main-qb` |
| `rbln` | `unified-sdk:main-rbln` | `main-rbln` |
| `warboy` | `unified-sdk:main-warboy` | `main-warboy` |
| `rngd` | `unified-sdk:main-rngd` | `main-rngd` |
| `trt --flavor vision` | `unified-sdk:main-trt-vision` | `main-trt-vision` |
| `trt --flavor llm` | `unified-sdk:main-trt-llm` | `main-trt-llm` |

## 사용 예시

### 권장 사용 흐름

처음 `main`을 검증할 때는 아래 순서를 권장합니다.

1. 원하는 backend용 Docker를 빌드합니다.
2. 빌드 스크립트가 출력한 `docker run ...` 예시로 컨테이너에 진입합니다.
3. 기본 sanity check를 먼저 실행합니다.
4. example `--help`를 확인합니다.
5. 최소 smoke entry를 실행합니다.

이 순서는 문제를 다음 세 층으로 분리하기 쉽게 해줍니다.

- Docker / device / toolkit 환경 문제
- Python dependency / import 문제
- 실제 vendor compile / runtime 문제

### 1차 검증 순서

VM별 1차 검증 권장 순서:

1. `qb`
2. `rbln`
3. `warboy`
4. `rngd`
5. `trt --flavor vision`
6. `trt --flavor llm`

이 순서를 권장하는 이유는,
상대적으로 단순한 vision/runtime 경로부터 확인한 뒤
LLM 및 flavor 분리 경로로 점차 확장하는 편이 문제 원인을 분리하기 쉽기 때문입니다.

### backend별 기본 sanity check

`qb`

```bash
python3 -c "import unified_sdk, qbruntime; print('OK')"
python3 examples/run_qb_build.py --help
python3 examples/run_qb_llm_infer.py --help
```

`rbln`

```bash
python3 -c "import unified_sdk, rebel; print('OK')"
python3 examples/run_rbln_build.py --help
python3 examples/run_rbln_llm_infer.py --help
```

`warboy`

```bash
python3 -c "import unified_sdk; print('OK')"
python3 examples/run_warboy_build.py --help
python3 examples/run_warboy_infer.py --help
```

`rngd`

```bash
python3 -c "import unified_sdk; print('OK')"
python3 examples/run_rngd_build.py --help
python3 examples/run_rngd_infer.py --help
```

`trt vision`

```bash
python3 -c "import unified_sdk; print('OK')"
python3 examples/run_tensorrt_build.py --help
python3 examples/run_tensorrt_infer.py --help
```

`trt llm`

```bash
python3 -c "import unified_sdk; print('OK')"
python3 examples/run_tensorrt_llm_build.py --help
python3 examples/run_tensorrt_llm_infer.py --help
```

### 1차 smoke entry

아래는 바로 복붙해서 보기 좋은 최소 smoke 예시입니다.  
모든 예제가 모든 환경에서 완전히 같은 수준으로 보장되는 것은 아니므로,
README와 branch별 known issue를 함께 확인하는 것을 권장합니다.

`qb`

```bash
./build.sh --backend qb
python3 examples/run_qb_build.py \
  --model-name resnet50 \
  --input-name input \
  --input-shape 1,3,224,224

python3 examples/run_qb_infer.py \
  --engine-path builds/resnet50.mxq \
  --image models/input.jpg
```

`rbln`

```bash
./build.sh --backend rbln
python3 examples/run_rbln_build.py \
  --model-zoo-model resnet50 \
  --pretrained \
  --model-name resnet50

python3 examples/run_rbln_infer.py \
  --engine-path builds/resnet50.rbln \
  --image models/input.jpg
```

LLM runtime smoke:

```bash
python3 examples/run_rbln_llm_infer.py \
  --engine-path Qwen/Qwen3-0.6B \
  --prompt "What is the capital of South Korea?"
```

`warboy`

```bash
./build.sh --backend warboy
python3 examples/run_warboy_build.py \
  --model-name resnet50 \
  --input-name input \
  --input-shape 1,3,224,224

python3 examples/run_warboy_infer.py \
  --engine-path builds/resnet50.enf \
  --image models/input.jpg
```

`rngd`

```bash
./build.sh --backend rngd
python3 examples/run_rngd_build.py \
  --model furiosa-ai/Qwen2.5-0.5B-Instruct

python3 examples/run_rngd_infer.py \
  --engine-path furiosa-ai/Qwen2.5-0.5B-Instruct \
  --prompt "What is the capital of South Korea?"
```

`trt vision`

```bash
./build.sh --backend trt --flavor vision
python3 examples/run_tensorrt_build.py \
  --model-name resnet50 \
  --precision fp32 \
  --input-name input \
  --input-shape 1,3,224,224

python3 examples/run_tensorrt_infer.py \
  --engine-path build_output/resnet50_FP32.engine \
  --input-name input \
  --output-name output \
  --input-shape 1,3,224,224
```

`trt llm`

```bash
./build.sh --backend trt --flavor llm
python3 examples/run_tensorrt_llm_build.py \
  --model-ref TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode fetch

python3 examples/run_tensorrt_llm_infer.py \
  --engine-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"
```

## 라이선스

본 프로젝트는 Apache License 2.0 하에 배포됩니다.

- 상업적 사용, 수정 및 재배포가 허용됩니다.
- 본 SDK는 기존 NPU 벤더 SDK 위에서 동작하는 통합 추상화 계층을 제공합니다.
- 각 백엔드 플러그인은 해당 NPU 벤더 SDK에 의존하며, 해당 SDK의 라이선스 및 지식재산권 정책을 따릅니다.

자세한 내용은 `LICENSE` 파일을 참고하십시오.

## 참고사항

- 이 프로젝트는 vendor compiler/runtime 자체를 다시 구현하지 않습니다.
- Unified SDK는 vendor SDK를 감싸는 공통 API 계층입니다.
- backend별 제약은 README, example, report에 그대로 남겨 추적합니다.
- 완전한 기능 동등성보다 **공통 API 이름과 공통 사용 흐름의 정리**를 우선합니다.

관련 참고 자료:

- `vendor_sdk_wrapping_api_report.md`
  - backend별 public API mapping, 구현 수준, known issue 요약
- `main_integration_review_2026-07-27.md`
  - historical merge와 post-merge normalization 방향 정리
- 각 vendor branch README
  - backend별 세부 smoke, vendor caveat, known issue의 원본 기록
