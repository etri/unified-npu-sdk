# Micro_DC 워크스페이스 진단 보고서

- 작성일: 2026-05-28
- 대상: `Micro_DC/` (git worktree 기반 멀티 백엔드 레이아웃)
- 작성: Claude (대화형 분석)

---

## 1. 워크스페이스 구조

```
Micro_DC/
├── .agents/  .codex/  .git/   (비어있음 — 무관 placeholder)
├── main/           [branch: main]          ← 메인 worktree (TRT + RBLN)
├── trt-only/       [branch: trt-only]      ← TensorRT 전용
├── rbln-only/      [branch: rbln-only]     ← RBLN 전용 (구축 대상)
├── qb-only/        [branch: qb-only]       ← README만 (개발 대기)
└── furiosa-only/   [branch: furiosa-only]  ← README만 (개발 대기)
```

- 모든 하위 폴더는 `main`의 git worktree (`.git`이 디렉토리가 아니라 `gitdir:` 포인터 파일)
- 원격: `https://github.com/etri/unified-npu-sdk.git`
- 브랜치명 규칙: 하이픈 통일 (`rbln-only` 포함, 2026-05-28 정정 완료)

---

## 2. 해결된 이슈 (이전 평가 → 현재)

| 항목 | 이전 상태 | 해결 커밋/조치 |
|---|---|---|
| `builder/` ↔ `build` 폴더명 불일치로 import 실패 | Critical | commit `a1d569c` 폴더 rename |
| 예제 파일들의 `import sys` 누락 | Critical | commit `43c780c` 전 예제 추가 |
| `jskang@rebellion/` 빈 nested repo | 구조 문제 | worktree 체제로 흡수 |
| 브랜치명 표기 혼선 (`rbln_only` vs `rbln-only`) | 일관성 | 원격/로컬 모두 hyphen 통일 |
| stale 로컬 브랜치 (`aris_only`, `rngd-only`, `rngd_only`) | cleanup | 삭제 완료 |

---

## 3. 남아있는 이슈

### 🔴 Critical
1. **`Dockerfiles/tensorrt/requirements.txt` 빈 파일 (0 bytes)**
   - 경로: `main/Dockerfiles/tensorrt/requirements.txt`
   - Docker 빌드는 통과하지만 의도된 의존성이 손실됨
   - `trt-only/requirements.txt`도 2바이트(개행만)

2. **`build.sh`가 존재하지 않는 Dockerfile을 가리킴**
   - `main/build.sh:102` → `Dockerfiles/rebellions/Dockerfile`
   - `main/build.sh:108` → `Dockerfiles/furiosa/Dockerfile`
   - 실제로는 `tensorrt/`만 존재. `--target rebellions`/`furiosa` 호출 시 즉시 실패
   - 백엔드별 worktree 체제 도입 이후 멀티타겟 분기는 제거하거나 worktree로 위임 필요

### 🟡 일관성/품질
3. **백엔드 자동 등록 코드 중복**
   - 동일 try/except import 블록이 `build/__init__.py:16-28`과 `build/api.py:11-23` 양쪽 존재
   - runtime 쪽도 동일. 한쪽에만 두면 충분

4. **`pyrightconfig.json` ↔ `pyproject.toml` Python 버전 불일치**
   - pyright: `"pythonVersion": "3.8"`
   - pyproject: `requires-python = ">=3.9"`

5. **`types.py` mutable default 타입 거짓말**
   - `extra: Dict[str, Any] = None`은 `Optional[Dict[str, Any]] = None`이어야

6. **TensorRT 런타임 `destroy`에서 GPU 메모리 해제 보장 X**
   - `runtime/tensorrt_runtime.py:120-126`: `del ctx[k]`만 — PyCUDA `mem_alloc`은 GC 의존
   - 명시적 `free()` 권장

7. **`devcontainer.json` 특정 개발자 경로 하드코딩**
   - `/home/etri/users/rskim/uDC` — 다른 개발자는 즉시 사용 불가

### 🟢 정리 권장
8. **`trt-only/` 백업 스크립트 그대로 커밋됨**
   - `build_BK.sh`, `build_origin.sh` — 의도 불명

9. **`main/src/unified_sdk/__init__.py` 모듈명 표기 불일치**
   - 주석은 `build:` (정상) — 폴더 rename 이후 해소됨. 단, README 구조도와 미세 차이 점검 필요

---

## 4. 단일-백엔드 worktree 패턴 (적용 중)

`trt-only`가 레퍼런스. 핵심 규칙:

- `types.py`의 `BuildBackendName` / `RuntimeBackendName` Literal을 단일 백엔드로 축소
- `build/__init__.py`와 `runtime/__init__.py`는 try/except 제거, 단일 어댑터 직접 import
- 루트에 단일 `Dockerfile` (main의 `Dockerfiles/<target>/` 구조 미사용)
- examples는 해당 백엔드 것만

이 패턴을 `rbln-only`에도 적용 (별도 작업 진행 중, 2026-05-28).

---

## 5. 권장 후속 작업 우선순위

1. `Dockerfiles/tensorrt/requirements.txt` 의존성 채우기
2. `main/build.sh`에서 미존재 타겟(`rebellions`, `furiosa`) 분기 제거 (worktree로 위임)
3. `devcontainer.json` 경로 일반화 (환경변수 또는 상대경로)
4. `types.py` Optional 타입 정정 + 중복 import 블록 정리
5. TRT 런타임 `destroy` 명시적 메모리 해제
