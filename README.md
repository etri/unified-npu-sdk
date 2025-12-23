# Unified SDK

**Unified SDK**는 PyTorch, TensorFlow 등의 AI 모델을 다양한 국산 AI 반도체(NPU) 환경에서 손쉽게 **컴파일 및 추론(런타임 실행)** 할 수 있도록 지원하는 통합 개발 도구입니다.

## 📘 프로젝트 개요

본 프로젝트는 **「국산 AI 반도체 기반 마이크로 데이터센터 확산 사업」** 내
**(세부 3) 국산 AI 반도체 기반 마이크로 데이터센터 운영 및 확산 기술 개발 과제**에서 수행한
**이종 AI 반도체 활용을 지원하는 통합 SDK** 결과물입니다.

이 SDK는 다양한 AI 반도체(TensorRT, 리벨리온, 퓨리오사 등) 간의 **추론 환경 통합**을 목표로 하며,
AI 모델의 빌드(컴파일) 및 런타임 생성 기능을 제공합니다.

---

## 🚀 주요 기능

| 구분                 | 설명                                        |
| ------------------ | ----------------------------------------- |
| 🧩 모델 컴파일(Build)   | PyTorch, TensorFlow 모델을 각 백엔드용 실행 파일로 컴파일 |
| ⚙️ 런타임 생성(Runtime) | 컴파일된 모델 파일을 로드하여 추론 엔진 실행                 |
| 🔌 백엔드 확장          | TensorRT, 리벨리온, 퓨리오사 등 국산 NPU 지원 예정       |

---

## 🏗️ 프로젝트 구조
```
unified-sdk/
├── README.md
├── requirements.txt
├── Dockerfile
├── devcontainer.json
├── examples/                     # 예제 코드
│   ├── test_tensorrt_build.py    # TensorRT 모델 빌드 예제
│   └── run_tensorrt_infer.py     # TensorRT 엔진 실행 예제
└── src/
    └── unified_sdk/
        ├── __init__.py
        ├── types.py              # 공통 데이터 구조 정의
        │
        ├── build/                # 모델 빌드(컴파일) 관련 모듈
        │   ├── __init__.py
        │   ├── api.py            # 상위 진입점 (backend-agnostic)
        │   ├── registry.py       # 백엔드 빌더 등록/조회 관리
        │   └── tensorrt_build.py # TensorRT 빌드 어댑터
        │
        └── runtime/              # 모델 실행(추론) 관련 모듈
            ├── __init__.py
            ├── api.py            # 상위 진입점 (backend-agnostic)
            ├── registry.py       # 백엔드 런타임 등록/조회 관리
            └── tensorrt_rt.py    # TensorRT 런타임 어댑터
```
---

## 💾 설치 방법

아래 명령어로 프로젝트를 로컬 개발 모드로 설치할 수 있습니다:

```bash
pip install -e .
```

---


