# Unified SDK

Unified SDK는 여러 AI 가속기 백엔드를 하나의 public API 골격으로 묶는 통합 SDK입니다.  
`main` 브랜치는 각 vendor branch의 이력을 보존 merge한 뒤, 공통 진입점과 사용 흐름을 다시 정리해가는 통합 작업 브랜치입니다.

## 프로젝트 상태

- `main`은 단일 안정 배포본이라기보다 **멀티밴더 통합 진행 브랜치**입니다.
- branch별 history import는 완료되었습니다.
- 현재는 `./build.sh --backend ...` dispatcher와 공통 Unified SDK public API 골격을 기준으로 재정비 중입니다.
- backend별 지원 수준은 아직 다르며, `planned` / `unsupported` / `known issue`를 그대로 유지합니다.

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

## 빠른 시작

레포 클론:

```bash
git clone https://github.com/etri/unified-npu-sdk.git
cd unified-npu-sdk/main
```

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

## 참고

- 이 프로젝트는 vendor compiler/runtime를 다시 구현하지 않습니다.
- Unified SDK는 vendor SDK를 감싸는 공통 API 계층입니다.
- backend별 제약은 README, example, report에 그대로 남겨 추적합니다.

## 라이선스

Apache License 2.0

