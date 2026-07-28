# Unified SDK

Unified SDK는 여러 AI 가속기 백엔드를 하나의 public API 골격으로 묶는 통합 SDK입니다.  
`main` 브랜치는 각 vendor branch의 이력을 보존 merge한 뒤, 공통 진입점과 사용 흐름을 다시 정리해가는 통합 작업 브랜치입니다.

이 저장소의 목적은 특정 벤더 SDK를 대체하는 것이 아니라, 각 벤더가 제공하는 compiler/runtime를 **공통 API와 공통 작업 흐름** 위에 올려 두는 것입니다.  
즉 사용자는 backend마다 완전히 다른 스크립트를 외우기보다, 가능한 한 비슷한 build / runtime / infer 흐름으로 접근하고, backend별 차이는 adapter와 문서에서 확인하는 방식을 목표로 합니다.

## 프로젝트 상태

- `main`은 단일 안정 배포본이라기보다 **멀티밴더 통합 진행 브랜치**입니다.
- branch별 history import는 완료되었습니다.
- 현재는 `./build.sh --backend ...` dispatcher와 공통 Unified SDK public API 골격을 기준으로 재정비 중입니다.
- backend별 지원 수준은 아직 다르며, `planned` / `unsupported` / `known issue`를 그대로 유지합니다.

## 이 브랜치를 어떻게 읽으면 좋은가

`main`은 “모든 backend가 완전히 동일하게 동작하는 최종 제품”이라기보다, 아래 세 층을 하나로 묶는 통합 작업 공간입니다.

1. vendor별로 검증된 branch 이력 보존
2. 공통 Unified SDK API 이름 정렬
3. backend별 Docker / smoke / example 진입점 통합

그래서 현재 문서는 다음 관점으로 읽는 것이 좋습니다.

- **공통 골격**: 어떤 public API 이름을 기준으로 맞춰가고 있는가
- **backend별 차이**: 어디까지 구현됐고, 어떤 부분이 planned / unsupported / known issue 인가
- **실행 흐름**: 어떤 Docker flavor와 어떤 example부터 검증하면 되는가

## 설계 원칙

- public API 이름은 가능한 한 통일합니다.
- vendor SDK의 제약은 숨기지 않고 문서와 example에 그대로 남깁니다.
- Docker 환경은 필요 시 vendor별, track별로 분리합니다.
- compile / runtime / inspect / example 흐름은 backend별로 최대한 비슷한 형태로 배치합니다.
- 완전한 기능 동등성보다 **구조 일관성**을 먼저 맞춥니다.

## 현재 포함된 backend

| Backend | Track | Docker | 현재 상태 |
| --- | --- | --- | --- |
| `qb` | vision + low-level LLM runtime | unified | vision 동작, LLM runtime preview, LLM build planned |
| `rbln` | vision + LLM | unified | API 구현 완료, container compile known issue 메모 유지 |
| `warboy` | vision | unified | build / infer / inspect 흐름 구현 |
| `rngd` | LLM | unified | generate 경로 구현, 일부 `fxb build`는 vendor toolchain 이슈 이력 있음 |
| `trt` | vision / LLM | split flavor | `vision`, `llm` 분리 Docker, LLM compile 일부 unsupported |

## 공통 public API

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

- API 이름은 통일하지만, backend별 semantics는 아직 완전히 동일하지 않습니다.
- 예: `qb` LLM은 high-level text generation보다 low-level cache-aware infer 성격이 강합니다.

## main에서 기대할 수 있는 것

현재 `main`에서 우선 기대할 수 있는 것은 아래입니다.

- backend별 build dispatcher 진입점
- 공통 Unified SDK public API 골격
- vendor/track별 smoke example 집합
- backend별 현재 구현 상태와 known issue를 한 곳에서 확인할 수 있는 문서

반대로 아직 backend마다 차이가 남아 있는 부분도 있습니다.

- LLM build 지원 수준
- container compile 안정성
- artifact 저장 방식
- high-level generate 와 low-level infer의 semantic 차이

즉 `main`은 “공통 이름과 공통 흐름”을 먼저 제공하고, backend별 세부 제약은 그대로 노출하는 방식으로 운영됩니다.

## 빠른 시작

레포 클론:

```bash
git clone https://github.com/etri/unified-npu-sdk.git
cd unified-npu-sdk/main
```

이미 vendor별 checkout이 `~/unified-npu-sdk/qb-only`, `~/unified-npu-sdk/rbln-only`처럼 따로 있는 경우에는,
상위 디렉터리에서 `main` 브랜치만 별도 하위 폴더로 받는 방식이 더 안전합니다.

```bash
cd ~/unified-npu-sdk
git clone --branch main --single-branch https://github.com/etri/unified-npu-sdk.git main
cd main
```

메모:

- 위 명령은 `~/unified-npu-sdk/main` 폴더를 새로 생성합니다.
- `~/unified-npu-sdk/main` 폴더가 이미 있으면 clone이 실패할 수 있으니, 그 경우에는 폴더명을 바꾸거나 비운 뒤 다시 시도합니다.

## 권장 사용 흐름

처음 `main`을 검증할 때는 아래 순서를 권장합니다.

1. 원하는 backend용 Docker를 빌드합니다.
2. 빌드 스크립트가 출력한 `docker run ...` 예시로 컨테이너에 진입합니다.
3. README의 기본 sanity check를 먼저 실행합니다.
4. 해당 backend의 example `--help`를 확인합니다.
5. 그 다음 최소 smoke entry를 실행합니다.

이 순서를 권장하는 이유는, 문제를 만났을 때 원인을 다음 세 층으로 빠르게 분리할 수 있기 때문입니다.

- Docker / device / toolkit 환경 문제
- Python dependency / import 문제
- 실제 vendor compile / runtime 문제

### backend별 Docker build

```bash
./build.sh --backend qb
./build.sh --backend rbln
./build.sh --backend warboy
./build.sh --backend rngd
./build.sh --backend trt --flavor vision
./build.sh --backend trt --flavor llm
```

각 backend script는 빌드가 끝나면 해당 환경에 맞는 `docker run ...` 예시를 직접 출력합니다.

## 1차 검증 순서

VM별로 1차 검증할 때는 아래 순서를 권장합니다.

1. `qb`
2. `rbln`
3. `warboy`
4. `rngd`
5. `trt --flavor vision`
6. `trt --flavor llm`

권장 이유:

- `qb`, `warboy`는 비교적 단순한 vision/runtime 경로부터 확인 가능합니다.
- `rbln`은 장치/CDI와 container compile 제약을 같이 확인해야 합니다.
- `rngd`는 LLM runtime과 `fxb` 경로를 분리해서 보는 편이 좋습니다.
- `trt`는 `vision`과 `llm` Docker flavor가 다르므로 마지막에 분리 검증하는 편이 안전합니다.

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

## 1차 smoke entry

아래는 VM 검증 때 먼저 보기 좋은 최소 smoke entry입니다.

`qb`

```bash
./build.sh --backend qb
python3 examples/run_qb_build.py --help
python3 examples/run_qb_infer.py --help
python3 examples/run_qb_llm_infer.py --help
```

`rbln`

```bash
./build.sh --backend rbln
python3 examples/run_rbln_build.py --help
python3 examples/run_rbln_infer.py --help
python3 examples/run_rbln_llm_build.py --help
python3 examples/run_rbln_llm_infer.py --help
```

`warboy`

```bash
./build.sh --backend warboy
python3 examples/run_warboy_build.py --help
python3 examples/run_warboy_infer.py --help
```

`rngd`

```bash
./build.sh --backend rngd
python3 examples/run_rngd_build.py --help
python3 examples/run_rngd_infer.py --help
```

`trt vision`

```bash
./build.sh --backend trt --flavor vision
python3 examples/run_tensorrt_build.py --help
python3 examples/run_tensorrt_infer.py --help
```

`trt llm`

```bash
./build.sh --backend trt --flavor llm
python3 examples/run_tensorrt_llm_build.py --help
python3 examples/run_tensorrt_llm_infer.py --help
```

## 예제 진입점

예제는 “모든 backend가 동일한 명령을 쓴다”기보다, **동일한 역할의 예제가 비슷한 이름으로 존재하는 구조**를 목표로 정리되어 있습니다.

예를 들어:

- vision build 예제
- vision infer 예제
- LLM build 예제
- LLM infer/generate 예제
- inspect 예제

를 backend별로 대응시켜 읽으면 현재 지원 수준을 빠르게 파악할 수 있습니다.

Vision 예제:

- `examples/run_qb_build.py`
- `examples/run_warboy_build.py`
- `examples/run_rbln_build.py`
- `examples/run_tensorrt_build.py`
- `examples/run_*_infer.py`

LLM 예제:

- `examples/run_qb_llm_infer.py`
- `examples/run_rngd_build.py`
- `examples/run_rngd_infer.py`
- `examples/run_rbln_llm_build.py`
- `examples/run_rbln_llm_infer.py`
- `examples/run_tensorrt_llm_build.py`
- `examples/run_tensorrt_llm_infer.py`

## 디렉터리 구조

```text
main/
├── Dockers/                  # backend/flavor별 Dockerfiles + requirements
├── scripts/                  # backend별 build launcher
├── examples/                 # vendor/track별 smoke examples
└── src/unified_sdk/
    ├── build/
    ├── runtime/
    └── types.py
```

## 관련 참고 자료

이 브랜치 외에도 아래 자료를 같이 보면 현재 통합 방향을 이해하기 쉽습니다.

- `vendor_sdk_wrapping_api_report.md`
  - backend별 public API mapping, 구현 수준, known issue 요약
- `main_integration_review_2026-07-27.md`
  - historical merge와 post-merge normalization 방향 정리
- 각 vendor branch README
  - backend별 세부 smoke, vendor caveat, known issue의 원본 기록

즉 `main` README는 통합 개요와 진입점 중심으로 보고, 세부 제약은 branch별 문서와 report 문서를 같이 참조하는 구성이 적절합니다.

## 참고

- 이 프로젝트는 vendor compiler/runtime를 다시 구현하지 않습니다.
- Unified SDK는 vendor SDK를 감싸는 공통 API 계층입니다.
- backend별 제약은 README, example, report에 그대로 남겨 추적합니다.

## 라이선스

Apache License 2.0
