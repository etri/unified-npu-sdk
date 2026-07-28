# Smoke Test Full Description

작성일: 2026-07-28

이 문서는 `main` 브랜치에서 제공하는 **전체 smoke test 흐름**을 backend별로 정리한 실행 가이드입니다.
짧은 진입 방법은 `README.md`를 먼저 보고, 실제로 build / fetch / infer / inspect를 순서대로 점검할 때는 이 문서를 기준으로 따라가는 것을 권장합니다.

## 1. 문서 목적

이 문서는 다음 목적을 가집니다.

- `main` 브랜치에서 제공하는 smoke test 전체 범위를 한 번에 확인
- backend별로 **compile / fetch / infer / inspect** 경로를 구분
- Unified SDK public API와 실제 example script의 대응 관계를 정리
- 현재 가능한 항목, `planned`, `unsupported`, `known issue`를 함께 명시

## 2. 공통 API 개요

### Vision API

- `build_unified(cfg)`
- `create_runtime(cfg)`
- `infer(...)`
- `destroy_runtime(rh)`

### LLM API

- `build_unified_LLM(cfg)`
- `create_runtime_LLM(cfg)`
- `infer_LLM(...)`
- `generate_LLM(...)`
- `destroy_runtime_LLM(rh)`

메모:

- API 이름은 공통이지만 backend별 동작 의미는 동일하지 않습니다.
- 특히 `qb` LLM은 high-level text generation보다 **low-level cache-aware infer primitive**에 가깝습니다.

## 3. smoke 분류 기준

이 문서에서는 smoke를 아래 범주로 나눕니다.

- `표준 fetching`
  - vendor model zoo / 공식 HF model id / 공식 artifact ref를 이용하는 기본 경로
- `custom fetching`
  - 사용자가 이미 준비한 artifact를 그대로 읽는 경로
- `custom compile`
  - ONNX, PTH/PT, local model path 등으로부터 새 artifact를 생성하는 경로
- `infer / generate`
  - 실제 runtime path를 확인하는 경로
- `inspect`
  - artifact metadata 또는 runtime load 가능 여부를 확인하는 경로

## 4. backend별 지원 범위 요약

| Backend | Vision | LLM | 현재 핵심 smoke |
| --- | --- | --- | --- |
| `qb` | 구현 | 부분 구현 | vision build/infer + low-level LLM infer |
| `rbln` | 구현 | 구현 | vision build/infer + LLM build/generate |
| `warboy` | 구현 | N/A | ENF fetch/compile + infer + inspect |
| `rngd` | N/A | 구현 | model id / FXB / local model generate |
| `trt` | 구현 | 구현 | TensorRT vision build/infer + TRT-LLM generate |

## 5. 공통 실행 원칙

### 5-1. Docker 진입

각 backend는 먼저 아래 dispatcher로 이미지 빌드를 수행합니다.

```bash
./build.sh --backend qb
./build.sh --backend rbln
./build.sh --backend warboy
./build.sh --backend rngd
./build.sh --backend trt --flavor vision
./build.sh --backend trt --flavor llm
```

정확한 `docker run ...` 명령은 각 build script가 출력하는 예시를 우선 사용합니다.

### 5-2. 기본 sanity check

각 backend는 smoke 실행 전에 다음 수준을 먼저 확인합니다.

1. 장치/드라이버/toolkit 인식
2. `python3 -c "import unified_sdk; ..."` import
3. example script `--help`

## 6. QB smoke

### 6-1. 개요

- artifact: `.mxq`
- vision build/infer는 구현
- LLM은 **precompiled transformer/LLM `.mxq` + low-level `infer_LLM(...)` smoke** 중심
- `build_unified_LLM(cfg)`는 현재 `planned`

### 6-2. 관련 example

- `examples/run_qb_build.py`
- `examples/run_qb_infer.py`
- `examples/inspect_qb_model.py`
- `examples/prepare_qb_transformer_model.py`
- `examples/run_qb_llm_infer.py`
- `examples/inspect_qb_llm_model.py`
- `examples/generate_qb_llm.py` (preview helper)

### 6-3. Vision smoke

#### 6-3-a. 표준 fetching

```bash
python3 examples/run_qb_build.py \
  --model-name resnet50 \
  --input-name input \
  --input-shape 1,3,224,224
```

#### 6-3-b. custom fetching

```bash
python3 examples/run_qb_build.py \
  --mxq models/resnet50.mxq \
  --model-name resnet50
```

#### 6-3-c. custom compile

ONNX:

```bash
python3 examples/run_qb_build.py \
  --from-onnx models/resnet50.onnx \
  --model-name resnet50 \
  --input-name input \
  --input-shape 1,3,224,224
```

PTH/PT:

```bash
python3 examples/run_qb_build.py \
  --from-pth models/resnet50.pth \
  --model-name resnet50 \
  --input-name input \
  --input-shape 1,3,224,224
```

#### 6-3-d. infer

```bash
python3 examples/run_qb_infer.py \
  --engine-path builds/resnet50.mxq \
  --core-mode global8 \
  --image models/input.jpg
```

#### 6-3-e. inspect

```bash
python3 examples/inspect_qb_model.py builds/resnet50.mxq
```

### 6-4. LLM smoke

#### 6-4-a. transformer/LLM MXQ 준비

```bash
python3 examples/prepare_qb_transformer_model.py \
  --model-id mobilint/Llama-3.2-1B-Instruct
```

#### 6-4-b. low-level runtime smoke

```bash
python3 examples/run_qb_llm_infer.py \
  --engine-path models/Llama-3.2-1B-Instruct.mxq \
  --core-mode global8 \
  --cache-size 0
```

Batch LLM:

```bash
python3 examples/run_qb_llm_infer.py \
  --engine-path models/Llama-3.2-1B-Instruct.mxq \
  --core-mode global8 \
  --batch-seq-lens 10,80 \
  --iters 3
```

#### 6-4-c. inspect

```bash
python3 examples/inspect_qb_llm_model.py \
  models/Llama-3.2-1B-Instruct.mxq \
  --core-mode global8
```

#### 6-4-d. 상태 메모

- `qb` LLM smoke는 high-level `generate(text)` 완료 기준이 아닙니다.
- 현재 완료 기준은 **precompiled `.mxq` fetch + low-level cache-aware infer + inspect**입니다.
- `build_unified_LLM(cfg)`는 vendor compile contract 공개 범위 재검토 전까지 `planned` 상태입니다.

## 7. RBLN smoke

### 7-1. 개요

- artifact: `.rbln`
- vision / LLM public API 모두 구현
- host native compile은 비교적 잘 되지만, Docker/CDI container compile은 vendor backend 이슈 영향 가능

### 7-2. 관련 example

- `examples/run_rbln_build.py`
- `examples/run_rbln_infer.py`
- `examples/inspect_rbln_model.py`
- `examples/run_rbln_llm_build.py`
- `examples/run_rbln_llm_infer.py`
- `examples/inspect_rbln_llm_model.py`

### 7-3. Vision smoke

#### 7-3-a. 표준 fetching

```bash
python3 examples/run_rbln_build.py \
  --model-zoo-model resnet50 \
  --pretrained \
  --model-name resnet50
```

#### 7-3-b. custom fetching

```bash
python3 examples/run_rbln_build.py \
  --rbln builds/resnet50.rbln \
  --model-name resnet50
```

#### 7-3-c. custom compile

PTH/PT:

```bash
python3 examples/run_rbln_build.py \
  --from-pth models/resnet50.pth \
  --model-name resnet50_pth \
  --input-shape 1,3,224,224
```

ONNX:

```bash
python3 examples/run_rbln_build.py \
  --from-onnx models/resnet50.onnx \
  --model-name resnet50_onnx \
  --input-shape 1,3,224,224
```

#### 7-3-d. infer

```bash
python3 examples/run_rbln_infer.py \
  --engine-path builds/resnet50.rbln \
  --image models/input.jpg
```

#### 7-3-e. inspect

```bash
python3 examples/inspect_rbln_model.py builds/resnet50.rbln
```

### 7-4. LLM smoke

#### 7-4-a. model id -> generate

```bash
python3 examples/run_rbln_llm_infer.py \
  --engine-path Qwen/Qwen3-0.6B \
  --prompt "What is the capital of South Korea?"
```

#### 7-4-b. build/fetch artifact

```bash
python3 examples/run_rbln_llm_build.py \
  --model Qwen/Qwen3-0.6B \
  --build-mode fetch
```

또는:

```bash
python3 examples/run_rbln_llm_build.py \
  --model Qwen/Qwen3-0.6B \
  --build-mode optimum_compile \
  --model-name qwen3_0_6b_rbln \
  --max-model-len 512 \
  --num-devices 1
```

#### 7-4-c. inspect

```bash
python3 examples/inspect_rbln_llm_model.py Qwen/Qwen3-0.6B
```

#### 7-4-d. 상태 메모

- host native에서는 `1/2/3` 경로가 통과한 이력이 있습니다.
- Docker/CDI container에서는 compile backend 이슈 때문에 같은 경로가 실패할 수 있습니다.
- precompiled artifact를 써도 runtime warmup 중 vendor-side internal compile이 다시 발생할 수 있습니다.

## 8. Warboy smoke

### 8-1. 개요

- artifact: `.enf`
- vision 전용 branch
- 표준 fetch / provided `.enf` / quantized ONNX compile / PTH->ONNX->quantized ONNX 흐름 정리

### 8-2. 관련 example

- `examples/run_warboy_build.py`
- `examples/run_warboy_infer.py`
- `examples/inspect_warboy_model.py`
- `examples/prepare_warboy_quantized_onnx.py`

### 8-3. Vision smoke

#### 8-3-a. 표준 fetching

```bash
python3 examples/run_warboy_build.py --list-model-zoo
python3 examples/run_warboy_build.py --model-name resnet50
```

#### 8-3-b. custom fetching

```bash
python3 examples/run_warboy_build.py \
  --enf models/resnet50.enf \
  --model-name resnet50
```

#### 8-3-c. custom compile

plain ONNX:

```bash
python3 examples/run_warboy_build.py \
  --from-onnx models/resnet50_quantized.onnx \
  --model-name resnet50
```

PTH/PT -> quantized ONNX 준비:

```bash
python3 examples/prepare_warboy_quantized_onnx.py \
  --source pth \
  --weights models/resnet50.pth \
  --model-name resnet50

python3 examples/run_warboy_build.py \
  --from-onnx models/resnet50_quantized.onnx \
  --model-name resnet50
```

model-zoo 원본 ONNX 기반:

```bash
python3 examples/prepare_warboy_quantized_onnx.py \
  --source model-zoo \
  --model-name resnet50

python3 examples/run_warboy_build.py \
  --from-onnx models/resnet50_quantized.onnx \
  --model-name resnet50
```

#### 8-3-d. infer

```bash
python3 examples/run_warboy_infer.py \
  --engine-path builds/resnet50.enf \
  --image models/input.jpg
```

#### 8-3-e. inspect

```bash
python3 examples/inspect_warboy_model.py builds/resnet50.enf
```

#### 8-3-f. detection 예시

```bash
python3 examples/run_warboy_build.py --model-name yolov5l
python3 examples/inspect_warboy_model.py builds/yolov5l.enf
```

### 8-4. 상태 메모

- quantized ENF 입력 dtype/layout은 모델 스펙에 고정되므로, preprocess 정합이 중요합니다.
- `main`의 infer helper는 classification/detection smoke를 위해 uint8 fallback을 추가로 보강한 상태입니다.

## 9. RNGD smoke

### 9-1. 개요

- artifact: `.fxb`
- LLM 전용 branch
- 표준 경로는 `model id -> generate`
- custom 경로는 `local model path + explicit FXB`, `local model path -> fxb build`

### 9-2. 관련 example

- `examples/run_rngd_build.py`
- `examples/run_rngd_infer.py`
- `examples/inspect_rngd_model.py`
- `examples/prepare_rngd_local_model.py`
- `examples/prepare_rngd_compatible_fxb.py`

### 9-3. LLM smoke

#### 9-3-a. 표준 fetching

```bash
python3 examples/run_rngd_build.py --model furiosa-ai/Qwen2.5-0.5B-Instruct

python3 examples/run_rngd_infer.py \
  --engine-path furiosa-ai/Qwen2.5-0.5B-Instruct \
  --prompt "What is the capital of South Korea?"

python3 examples/inspect_rngd_model.py furiosa-ai/Qwen2.5-0.5B-Instruct
```

#### 9-3-b. custom fetching

local model snapshot 준비:

```bash
python3 examples/prepare_rngd_local_model.py \
  --model Qwen/Qwen3-8B-FP8
```

compatible FXB 준비:

```bash
python3 examples/prepare_rngd_compatible_fxb.py \
  --model furiosa-ai/Qwen3-8B-FP8
```

runtime:

```bash
python3 examples/run_rngd_infer.py \
  --engine-path models/Qwen3-8B-FP8 \
  --fxb artifacts/Qwen3-8B-FP8.fxb \
  --prompt "What is the capital of South Korea?"
```

#### 9-3-c. custom build

```bash
rm -rf ~/.cache/furiosa/compiler/

python3 examples/run_rngd_build.py \
  --model models/Qwen3-8B-FP8 \
  --fxb-build \
  --model-name qwen3_8b_fp8 \
  --tensor-parallel-size 8
```

#### 9-3-d. 상태 메모

- `Qwen3-8B-FP8` custom `fxb build`는 vendor toolchain 이슈 이력이 있습니다.
- `gcc-aarch64-linux-gnu` 및 추가 cross toolchain 계열 패키지 반영 후 재검증 중입니다.
- custom smoke 전에는 `~/.cache/furiosa/compiler/`를 비우는 것을 권장합니다.

## 10. TensorRT smoke

### 10-1. 개요

- artifact: `.engine`
- `vision`과 `llm`은 서로 다른 Docker flavor
- vision은 비교적 정리됨
- LLM은 official TensorRT-LLM release container(PyTorch backend) 기준

### 10-2. 관련 example

- `examples/run_tensorrt_build.py`
- `examples/run_tensorrt_infer.py`
- `examples/inspect_engine_io.py`
- `examples/run_tensorrt_llm_build.py`
- `examples/run_tensorrt_llm_infer.py`
- `examples/inspect_tensorrt_llm_model.py`

### 10-3. Vision smoke

#### 10-3-a. 표준 fetching

```bash
python3 examples/run_tensorrt_build.py \
  --model-name resnet50 \
  --precision fp32 \
  --input-name input \
  --input-shape 1,3,224,224
```

#### 10-3-b. custom fetching

```bash
python3 examples/run_tensorrt_build.py \
  --engine ./build_output/resnet50_FP32.engine \
  --model-name resnet50 \
  --precision fp32
```

#### 10-3-c. custom compile

ONNX:

```bash
python3 examples/run_tensorrt_build.py \
  --model-name yolov7 \
  --precision fp32 \
  --input-name images \
  --input-shape 1,3,640,640
```

PTH/PT:

```bash
python3 examples/run_tensorrt_build.py \
  --from-pth models/resnet50.pth \
  --model-name resnet50 \
  --precision fp32 \
  --input-name input \
  --input-shape 1,3,224,224
```

#### 10-3-d. infer

```bash
python3 examples/run_tensorrt_infer.py \
  --engine-path build_output/resnet50_FP32.engine \
  --input-name input \
  --output-name output \
  --input-shape 1,3,224,224
```

#### 10-3-e. inspect

```bash
python3 examples/inspect_engine_io.py build_output/resnet50_FP32.engine
```

### 10-4. LLM smoke

#### 10-4-a. model id -> generate

```bash
python3 examples/run_tensorrt_llm_build.py \
  --model-ref TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode fetch

python3 examples/run_tensorrt_llm_infer.py \
  --engine-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of South Korea?"
```

#### 10-4-b. local prebuilt artifact -> generate

```bash
python3 examples/run_tensorrt_llm_infer.py \
  --engine-path artifacts/tinyllama_trtllm \
  --prompt "What is the capital of South Korea?"
```

#### 10-4-c. local model path -> compile -> generate

```bash
python3 examples/run_tensorrt_llm_build.py \
  --model-ref TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --build-mode llm_api_compile \
  --model-name tinyllama_trtllm
```

#### 10-4-d. inspect

```bash
python3 examples/inspect_tensorrt_llm_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

#### 10-4-e. 상태 메모

- `7-c` 성격의 compile path는 current release container(PyTorch backend)에서 `save()` 미지원으로 `unsupported`입니다.
- `7-b`는 concept 상 독립 경로지만, 실제로는 기존 artifact가 있어야 검증 가능합니다.

## 11. 상태 요약

### 11-1. 현재 바로 보기 좋은 smoke

- `qb`: vision build/infer + LLM low-level infer
- `rbln`: vision build/infer + LLM generate
- `warboy`: ENF fetch/compile + infer + inspect
- `rngd`: model id generate + local FXB generate
- `trt`: vision build/infer + LLM model id generate

### 11-2. planned / unsupported / known issue

- `qb` LLM compile: `planned`
- `rbln` container compile: `known issue`
- `warboy` preprocess/dtype 정합: `known caveat`
- `rngd` custom `fxb build`: vendor toolchain 이력 존재
- `trt` LLM `llm_api_compile`: current official release container 기준 `unsupported`

## 12. 참고 자료

- `README.md`
- `vendor_sdk_wrapping_api_report.md`
- `main_integration_review_2026-07-27.md`
- 각 vendor branch README
  - `qb-only`
  - `rbln-only`
  - `furiosa-only`
  - `furiosa-llm-only`
  - `trt-only`
