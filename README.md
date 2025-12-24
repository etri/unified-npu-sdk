# Unified SDK

**Unified SDK**는 PyTorch, TensorFlow 등의 AI 모델을 다양한 국산 AI 반도체(NPU) 환경에서 손쉽게 **컴파일 및 추론(런타임 실행)** 할 수 있도록 지원하는 통합 개발 도구입니다.

Unified SDK is an integrated development toolkit that enables seamless model compilation (build) and inference (runtime execution) of AI models such as PyTorch and TensorFlow across heterogeneous Korean AI accelerators (NPUs).

> This README is provided in both **Korean and English**.  
> 본 문서는 **한국어와 영어**로 제공됩니다.

---

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
│   ├── run_tensorrt_build.py     # TensorRT 모델 빌드 예제
│   ├── run_tensorrt_infer.py     # TensorRT 엔진 실행 예제
│   ├── run_rbln_build.py         # Rbln 모델 빌드 예제
│   └── run_rbln_infer.py         # Rbln 엔진 실행 예제
└── src/
    └── unified_sdk/
        ├── __init__.py
        ├── types.py              # 공통 데이터 구조 정의
        │
        ├── builder/                # 모델 빌드(컴파일) 관련 모듈
        │   ├── __init__.py
        │   ├── api.py            # 상위 진입점 (backend-agnostic)
        │   ├── registry.py       # 백엔드 빌더 등록/조회 관리
        │   ├── tensorrt_build.py # TensorRT 빌드 어댑터
        │   └── rbln_build.py     # Rbln 빌드 어댑터
        │
        └── runtime/              # 모델 실행(추론) 관련 모듈
            ├── __init__.py
            ├── api.py            # 상위 진입점 (backend-agnostic)
            ├── registry.py       # 백엔드 런타임 등록/조회 관리
            ├── tensorrt_runtime.py  # TensorRT 런타임 어댑터
            └── rbln_runtime.py   # rbln 런타임 어댑터
```
---

## 💾 설치 방법

아래 명령어로 프로젝트를 로컬 개발 모드로 설치할 수 있습니다:

```bash
pip install -e .
```

---

## 📜 라이선스

본 프로젝트는 Apache License 2.0 하에 배포됩니다.

* 상업적 사용, 수정 및 재배포가 허용됩니다.
* 본 SDK는 기존 NPU 벤더 SDK 위에서 동작하는 통합 추상화 계층을 제공합니다.
* 각 백엔드 플러그인은 해당 NPU 벤더 SDK에 의존하며, 해당 SDK의 라이선스 및 지식재산권(IP) 정책을 따릅니다.

자세한 내용은 LICENSE 파일을 참고하십시오.

---

## 📌 참고사항

* 본 프로젝트는 컴파일러 자체를 구현하지 않으며, 기존 NPU 벤더에서 제공하는 SDK를 호출하는 상위 통합 SDK입니다.
* NPU별 모델 빌드 및 런타임 동작의 차이는 플러그인(어댑터) 내부에 캡슐화되어 있습니다.
* 새로운 NPU는 레지스트리 기반 플러그인 구조를 통해 상위 Unified API 수정 없이 확장 가능합니다.
* 향후 다양한 국산 NPU 백엔드가 추가될 예정입니다.

---


## 📘 Project Overview

This project is an outcome of the
“Micro Data Center (μDC) Expansion Project Based on Korean AI Accelerators”,
specifically under
(Subtask 3) Development of Operation and Deployment Technologies for Micro Data Centers.

The Unified SDK abstracts vendor-specific NPU SDKs and APIs through a
Unified API and plugin-based backend architecture,
allowing AI services to be developed and deployed using a single, backend-agnostic API,
regardless of the underlying NPU hardware.

---

## 🚀 Key Features

| Category                 | Description                                        |
| --------------------- | ----------------------------------------- |
|🧩 Model Build	        | Compile PyTorch and TensorFlow models into executable formats for each backend
|⚙️ Runtime Execution	| Load compiled models and execute inference via a unified runtime API
|🔌 Backend Extension	| Designed to support multiple NPUs such as TensorRT, Rebellions, and Furiosa

---

## 🏗️ Project Structure
```
unified-sdk/
├── README.md
├── requirements.txt
├── Dockerfile
├── devcontainer.json
├── examples/                     # Example scripts
│   ├── test_tensorrt_build.py    # TensorRT build example
│   ├── run_tensorrt_infer.py     # TensorRT inference example
│   ├── run_rbln_build.py         # Rbln build example
│   └── run_rbln_infer.py         # Rbln inference example
└── src/
    └── unified_sdk/
        ├── __init__.py
        ├── types.py              # Common data structures
        │
        ├── builder/              # Model build (compilation) modules
        │   ├── api.py            # Unified entry point
        │   ├── registry.py       # Backend builder registry
        │   ├── tensorrt_build.py # TensorRT build adapter
        │   └── rbln_build.py     # Rbln build adapter
        │
        └── runtime/              # Model execution (inference) modules
            ├── api.py            # Unified entry point
            ├── registry.py       # Backend runtime registry
            ├── tensorrt_runtime.py  # TensorRT runtime adapter
            └── rbln_runtime.py   # rbln runtime adapter
```
---

## 💾 Installation

Install the project in editable (development) mode:

```bash
pip install -e .
```

---


## 📜 LICENSE

This project is released under the Apache License 2.0.

* Commercial use, modification, and redistribution are permitted.
* This SDK provides a unified abstraction layer over existing vendor-specific NPU SDKs.
* Each backend plugin depends on the corresponding vendor SDK and is subject to the vendor’s own license terms and intellectual property rights.

For full license details, please refer to the LICENSEfile.

---

## 📌 NOTICE

* This project does not implement a compiler or code generation toolchain. Instead, it provides a unified SDK layer that invokes existing vendor-provided NPU SDKs.
* Backend-specific build and runtime behaviors are encapsulated within plugin adapters.
* New NPU backends can be integrated via a registry-based plugin mechanism without modifying the upper-level Unified API.
* Additional Korean NPU backends will be integrated in future releases.

---




