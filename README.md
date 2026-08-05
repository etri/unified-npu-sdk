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

### 현재 구현 상태

| 구분 | 현재 상태 |
| --- | --- |
| Vision API | `build_unified` / `create_runtime` / `infer` / `destroy_runtime` 구현 |
| LLM API | `N/A` |
| Vision compile | model-zoo/source fetch / provided `.enf` fetch / quantized ONNX compile / PTH->ONNX->quantized ONNX->`.enf` 구현 |
| Vision smoke | compile / infer / inspect 흐름이 비교적 안정적으로 정리됨 |

### 주요 이슈

- 이 브랜치는 Warboy vision 전용입니다. RNGD/LLM 경로는 `furiosa-llm-only` 브랜치에서 다룹니다.
- 실무상 이슈는 vendor compiler 제약보다 **quantized ONNX 준비**, label/후처리 자산 정리 쪽이 더 큽니다.

---

## 🏗️ 프로젝트 구조

```
<repo-root>/
├── README.md
├── LICENSE
├── pyproject.toml
├── pyrightconfig.json
├── Dockers/
│   ├── docker.warboy.unified
│   └── requirements.warboy.unified.txt
├── devcontainer.json
├── build.sh
├── scripts/
│   └── build_warboy.sh
├── examples/
│   ├── prepare_warboy_quantized_onnx.py  # model-zoo / plain ONNX / .pth/.pt -> quantized ONNX 준비
│   ├── run_warboy_build.py         # model zoo ENF fetch / provided .enf fetch / quantized ONNX→.enf 컴파일
│   ├── run_warboy_infer.py         # .enf 모델 추론 (furiosa.runtime)
│   └── inspect_warboy_model.py     # .enf 입출력 메타 확인
└── src/unified_sdk/
    ├── __init__.py
    ├── types.py                    # core build/runtime config + Warboy capability contract
    ├── options.py                  # typed backend options (target_npu / runtime device 등)
    ├── frontends/
    │   ├── __init__.py
    │   ├── types.py                # prepare/fetch request/result contract
    │   ├── prepare_warboy_source.py
    │   ├── resolve_warboy_build_request.py
    │   └── warboy_model_zoo.py
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

> `models/` 폴더는 gitignore 대상이라 clone 직후에는 비어 있거나 아예 없을 수 있습니다.
> 필요하면 직접 생성해서 모델 자산을 넣어야 합니다.

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
- 기본 컴파일 타깃은 `warboy-2pe` 입니다.
- 1 PE 환경에서는 `--target-npu warboy` 또는 `WarboyBuildOptions(target_npu="warboy")` 를 사용하세요.

### 4. Docker 빌드 & 실행

```bash
./build.sh
# 종료 후 안내되는 docker run 명령을 참고하여 컨테이너 실행
```

`./build.sh`는 `torch`와 `torchvision`을 같은 CPU wheel index에서 함께 설치하고, `numpy==1.24.4`,
`opencv-python-headless==4.10.0.84`도 미리 고정한 뒤, `warboy-jammy` APT suite와
`furiosa-runtime`, `furiosa-optimizer`, `furiosa-quantizer`, `furiosa-models` 등
이 브랜치가 실제로 사용하는 Furiosa Python 패키지를 `0.10.2` 축으로 이미지에 설치합니다. Furiosa pip 인덱스가
따로 필요하면 `FURIOSA_PIP_INDEX=... ./build.sh` 또는 `./build.sh --furiosa-pip-index <url>`로 지정합니다.

이 브랜치의 build 입력은 아래 둘만 지원합니다.

- 사전 컴파일된 `.enf`
- `furiosa-compiler`로 컴파일할 **quantized ONNX**

표준 fetch는 `furiosa-models` model zoo가 제공하는 ENF 바이너리를 그대로 사용하는 방식입니다.
`examples/run_warboy_build.py --model-name <name>`는 먼저 `models/`에서 `.enf`를 찾고,
없으면 `furiosa.models.vision` model zoo에서 대응 ENF를 받아 `models/<model>.enf`로 정규화합니다.

추가로 `furiosa-models`는 원본 ONNX(`origin`)와 calibration 범위(`tensor_name_to_range`)도 제공하므로,
`examples/prepare_warboy_quantized_onnx.py --source model-zoo`로 quantized ONNX를 만든 뒤
`--from-onnx`로 넘기는 경로도 사용할 수 있습니다.

`resnet50.pth` 같은 PyTorch weight 파일을 직접 쓰는 경로는 `--source pth`로 남겨두었지만,
bare `.pth/.pt`만으로는 계산 그래프를 복원할 수 없으므로, 기본 내장 예제 외의 아키텍처는
`--model-factory package.module:callable`로 모델 복원 함수를 함께 넘겨야 합니다.

예:

```bash
# model zoo 목록 확인
python3 examples/run_warboy_build.py --list-model-zoo

# 표준 fetch: Furiosa model zoo ENF를 바로 확보
python3 examples/run_warboy_build.py --model-name resnet50

# 또는 model zoo 원본 ONNX + calibration range를 사용해 quantized ONNX를 만든 뒤 컴파일
python3 examples/prepare_warboy_quantized_onnx.py --source model-zoo --model-name resnet50
python3 examples/run_warboy_build.py --from-onnx models/resnet50_quantized.onnx --model-name resnet50
```

컨테이너 실행 예시:

```bash
docker run -it --security-opt seccomp=unconfined \
  --name furiosa-only \
  --device /dev/npu0:/dev/npu0 \
  -w /workspace/unified-sdk \
  -v $(pwd):/workspace/unified-sdk \
  unified-sdk:warboy
```

> 위 명령은 **최소 예시**입니다.
> 실제 호스트에서는 `/dev/npu0_bar*`, `/dev/npu0ch*`, `/dev/npu0pe*`, `/dev/npu0_mgmt` 같은
> 추가 장치 노드와 `furiosactl` 마운트가 더 필요할 수 있습니다.
> 따라서 실제 실행 시에는 README 예시보다 `./build.sh`가 출력한 `docker run ...` 명령을 **우선 사용**하세요.

컨테이너 내부 점검:

```bash
cd /workspace/unified-sdk
furiosactl list && furiosactl info || true
furiosa-compiler --version || true
python3 -c "import unified_sdk; from furiosa.runtime import sync; print('OK')"
python3 -c "import torch, torchvision; print('torch=', torch.__version__, 'torchvision=', torchvision.__version__)"

# (선택) model-zoo API 확인
python3 -c "from furiosa.models import vision; print(hasattr(vision, 'ResNet50'))"
```

---

## 🚀 Backend Docker smoke

아래 흐름은 **Warboy 장치가 호스트에 잡혀 있는 단일 머신**에서 Docker로 `furiosa-only`
백엔드를 검증하는 표준 smoke 절차입니다. 추가 wrapper 계층 없이 Unified SDK의 Warboy adapter가
vendor SDK(`furiosa-compiler`/`furiosa.runtime`)를 직접 호출합니다.

```bash
# 1) 이미지 빌드
./build.sh

# 2) README 예시가 아니라 build.sh가 출력한 docker run 명령으로 컨테이너 진입

# 3) 컨테이너 내부에서 장치/패키지 확인
furiosactl list && furiosactl info || true
furiosa-compiler --version || true
python3 -c "import unified_sdk; from furiosa.runtime import sync; print('OK')"

# 4) 표준 fetching smoke (vendor model zoo ENF fetch)
python3 examples/run_warboy_build.py --list-model-zoo
python3 examples/run_warboy_build.py --model-name resnet50

# 참고: 표준 fetch / plain ONNX quantization smoke 는 우선 `--list-model-zoo`에 보이는
# 모델 계열을 기준으로 잡습니다. 목록 밖의 generic ONNX 는 현재 vendor quantizer 지원 범위에 따라
# 실패할 수 있으므로 표준 smoke 대상으로 보지 않습니다.

# 4-b) custom fetching smoke (provided .enf)
python3 examples/run_warboy_build.py --enf models/resnet50.enf --model-name resnet50

# 4-c-1) custom compile smoke: plain ONNX -> quantized ONNX -> .enf
#        표준 기준은 `--list-model-zoo`에 보이는 모델 계열의 floating ONNX 입니다.
#        아래 예시는 user-prepared `models/yolov5l.onnx`를 사용하는 경우입니다.
#        quantization 단계는 calibration 이미지가 필요합니다.
python3 examples/prepare_warboy_quantized_onnx.py \
  --source onnx \
  --onnx models/yolov5l.onnx \
  --model-name yolov5l \
  --calib-image models/input.jpg

python3 examples/run_warboy_build.py \
  --from-onnx models/yolov5l_quantized.onnx \
  --model-name yolov5l

# 4-c-2) custom compile smoke: PTH/PT -> ONNX export -> quantized ONNX -> .enf
#        기본 내장 예제는 resnet50 이고, 다른 아키텍처는 --model-factory 로 복원 함수를 넘깁니다.
python3 examples/prepare_warboy_quantized_onnx.py \
  --source pth \
  --weights models/resnet50.pth \
  --model-name resnet50 \
  --calib-image models/input.jpg

python3 examples/run_warboy_build.py \
  --from-onnx models/resnet50_quantized.onnx \
  --model-name resnet50

# 예: 다른 아키텍처를 local checkpoint로 준비할 때
# python3 examples/prepare_warboy_quantized_onnx.py \
#   --source pth \
#   --weights models/custom_model.pth \
#   --model-name custom_model \
#   --model-factory mypkg.models:create_model \
#   --calib-image models/input.jpg

# 5) (선택) 1 PE 환경 참고 예제
#    아래 명령은 builds/resnet50.enf 를 1PE용으로 다시 생성하므로,
#    기본 2PE smoke 흐름을 순서대로 따라가는 중에는 실행하지 않는 것을 권장합니다.
#    실제 1PE 환경이거나 1PE ENF를 의도적으로 만들 때만 참고하세요.
# python3 examples/run_warboy_build.py \
#   --from-onnx models/resnet50_quantized.onnx \
#   --target-npu warboy \
#   --model-name resnet50

# 6) .enf 추론
#    resnet50.enf 이고 furiosa-models 가 있으면 model-zoo preprocess/postprocess 를 우선 사용합니다.
#    tests/input.jpg가 없으면 synthetic 입력으로 런타임 경로를 검증합니다.
#    ENF 입력 계약을 자동 해석하지 못하면 명령이 fail-closed 로 멈추며,
#    이 경우 --input-dtype uint8 또는 --input-dtype float32 로 재시도합니다.
python3 examples/run_warboy_infer.py \
  --engine-path builds/resnet50.enf \
  --iters 50

# 예: resnet50.enf 가 UINT8 입력을 기대하는 환경에서 자동 해석이 실패할 때
python3 examples/run_warboy_infer.py \
  --engine-path builds/resnet50.enf \
  --iters 50 \
  --input-dtype uint8

# 7) 모델 메타 best-effort 확인
python3 examples/inspect_warboy_model.py builds/resnet50.enf

# 8) detection 계열 infer smoke (YOLOv5l)
#    4-c-1 에서 이미 builds/yolov5l.enf 를 준비했다는 전제입니다.
#    detection 모델은 classification helper/postprocess 계약과 다를 수 있으므로,
#    이 단계의 목적은 runtime load + infer path + raw output shape 확인입니다.
python3 examples/run_warboy_infer.py \
  --engine-path builds/yolov5l.enf \
  --input-shape 1,3,640,640 \
  --iters 20

# infer 뒤에 모델 메타를 best-effort 로 다시 확인
python3 examples/inspect_warboy_model.py builds/yolov5l.enf

# 필요하면 detection smoke 도 같은 방식으로 dtype override 를 줄 수 있습니다.
# python3 examples/run_warboy_infer.py \
#   --engine-path builds/yolov5l.enf \
#   --input-shape 1,3,640,640 \
#   --iters 20 \
#   --input-dtype uint8
```

예제 스크립트는 checkout root를 자동 탐지하므로 `/workspace/unified-sdk`,
`/workspace/unified-npu-sdk`, 또는 현재 repository root에서 모두 실행할 수 있습니다.

---

## 🚀 사용 예시

### Build / Runtime API 분리

`furiosa-only`는 build / runtime wrapping API를 **Warboy vision capability** 기준으로 구분하며,
실제로는 아래처럼 `furiosa-compiler`와 `furiosa.runtime` 경로에 매핑됩니다.

| 용도 | 단계 | Unified SDK | 내부 vendor |
| --- | --- | --- | --- |
| Vision `.enf` | 빌드 | `build_unified(cfg)` | provided `.enf` 복사 또는 `furiosa-compiler <quantized_onnx> -o <enf>` |
| Vision `.enf` | 생성 | `create_runtime(cfg)` | `furiosa.runtime.sync.create_runner(str(enf_path), device=...)` |
| Vision `.enf` | 추론 | `infer(rh, input_array)` | `runner.run([input_array])` |
| Vision `.enf` | 종료 | `destroy_runtime(rh)` | `runner.close()` / `runner.__exit__(...)` best-effort |

기본 원칙:
- public runtime surface는 `create_runtime` / `infer` / `destroy_runtime` 기준으로 유지합니다.
- build core는 **provided `.enf` 배치** 또는 **quantized ONNX -> `.enf` 컴파일**만 담당합니다.
- quantization은 build 내부가 아니라 `prepare_warboy_quantized_onnx.py`와 `frontends` helper가 담당하는 별도 prepare capability로 봅니다.

### 컴파일 (.enf 생성)

```python
from unified_sdk.build import build_unified
from unified_sdk.options import WarboyBuildOptions
from unified_sdk.types import BuildConfig

# (a) standard fetch from model zoo ENF
cfg = BuildConfig(
    backend="warboy",
    model_or_path="models/resnet50.enf",
    out_dir="builds",
    model_name="resnet50",
    input_name="input",
    input_shape=(1, 3, 224, 224),
)
result = build_unified(cfg)
print(result.compiled_model_path)

# (b) quantized ONNX -> .enf (furiosa-compiler)
cfg = BuildConfig(
    backend="warboy",
    model_or_path="models/resnet50_quantized.onnx",  # quantized ONNX 경로
    out_dir="builds",
    model_name="resnet50",
    input_name="input",
    input_shape=(1, 3, 224, 224),
    backend_options=WarboyBuildOptions(
        target_npu="warboy-2pe",
        target_ir="enf",
    ),
)
result = build_unified(cfg)
print(result.compiled_model_path)

# 1 PE 환경 예시:
#     backend_options=WarboyBuildOptions(target_npu="warboy")
```

### 추론

```python
import numpy as np
from unified_sdk.runtime import create_runtime, infer, destroy_runtime
from unified_sdk.options import WarboyRuntimeOptions
from unified_sdk.types import RuntimeConfig

cfg = RuntimeConfig(
    backend="warboy",
    engine_path="builds/resnet50.enf",
    input_name="input",
    output_name="output",
    input_shape=(1, 3, 224, 224),
    backend_options=WarboyRuntimeOptions(device=None),   # 예: "warboy(0)*2"
)
rh = create_runtime(cfg)
y = infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
destroy_runtime(rh)
```

참고:
- `model-zoo` 경로의 `ResNet50` quantized ONNX 는 현재 확인 기준 ONNX output 이 `ArgMax:0 [1]` 형태입니다.
- 따라서 `run_warboy_infer.py` 는 `vision.ResNet50().postprocess(...)`를 우선 시도하고,
  그게 불가능할 때만 `(1,)` scalar-like 출력을 top-1 class id 로 해석합니다.

---

## 📜 라이선스

Apache License 2.0. 자세한 내용은 LICENSE 파일 참조.
본 SDK는 FuriosaAI SDK(`furiosa-compiler`/`furiosa.runtime`/`furiosa.quantizer`) 위에서 동작하는 통합 추상화 계층이며, 해당 패키지의 라이선스/IP 정책을 따릅니다.

---

## 📌 참고

- 본 체크아웃은 Warboy 어댑터만 노출합니다. 다중 백엔드는 `main` 브랜치에서 사용하세요.
- `furiosa-compiler`는 ONNX(OpSet 13 이하)와 TFLite를 입력으로 받을 수 있지만, Warboy NPU 가속을 위해서는
  실질적으로 **quantized ONNX**를 준비하는 흐름을 사용합니다. Furiosa 공식 문서도 NPU 가속용으로 quantized model 사용을 권장합니다.
- `furiosa-models`는 Warboy용 open model zoo이며, `furiosa.models.vision` 또는 `furiosa-models list`로
  지원 모델을 확인할 수 있습니다. 현재 확인 가능한 대표 vision 모델은 `ResNet50`, `EfficientNetB0`,
  `EfficientNetV2s`, `SSDMobileNet`, `SSDResNet34`, `YOLOv5m`, `YOLOv5l`, `YOLOv7w6Pose` 입니다.
- detection/pose 계열 model zoo(`YOLOv5*`, `YOLOv7w6Pose`)는 내부적으로 `torchvision` postprocess 의존성을 사용할 수 있으므로,
  컨테이너에서는 `torch`/`torchvision`을 같은 PyTorch wheel index에서 함께 설치한 이미지를 기준으로 검증합니다.
- `models/` 디렉터리는 저장소에 포함되지 않을 수 있습니다(gitignore). 없으면 직접 만들면 됩니다.
- plain ONNX 는 `prepare_warboy_quantized_onnx.py --source onnx --onnx ...` 로 quantized ONNX 를 먼저 준비한 뒤
  `--from-onnx`로 넘깁니다. 이 경로의 표준 smoke 기준은 `--list-model-zoo`에 보이는 모델 계열이며,
  그 밖의 generic ONNX 는 vendor quantizer 지원 범위에 따라 실패할 수 있습니다.
- `.pth`/`.pt` 가중치 파일은 build 입력으로 직접 쓰지 않고, 필요하면 `prepare_warboy_quantized_onnx.py --source pth`
  로 quantized ONNX 를 먼저 준비한 뒤 `--from-onnx`로 넘깁니다. bare checkpoint만으로는 모델 구조를 복원할 수 없으므로,
  기본 예제 외 아키텍처는 `--model-factory package.module:callable` 같이 복원 함수를 함께 넘겨야 합니다.
- `.enf`의 입력 dtype/layout은 quantized ONNX 스펙에 따라 고정(보통 int8/uint8)되므로, 추론 입력을 이에 맞춰야 합니다.
- 다중 장치 서버에서는 `FURIOSA_DEVICES`/`--device`(예: `warboy(0)*2`)로 장치를 고정하세요.
- 장치/모델 점검용 CLI: `furiosactl list`, `furiosactl info`, `furiosa-smi info`.
- 예제 스크립트는 CLI 인자를 지원합니다. 자세한 옵션은 `python3 examples/run_warboy_build.py --help`,
  `python3 examples/run_warboy_infer.py --help`, `python3 examples/inspect_warboy_model.py --help`로 확인하세요.
- 다른 백엔드는 각 vendor 브랜치(`rbln-only`, `qb-only`, `furiosa-llm-only`)에서 작업하세요.
