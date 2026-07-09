# syntax=docker/dockerfile:1.7
# =========================
# unified-sdk (QB / Mobilint ARISE base)
# =========================
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_INPUT=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/workspace/unified-sdk/src

# 1) 기본 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git ca-certificates curl \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# 2) 사용자 생성
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/unified-sdk
RUN mkdir -p /workspace/unified-sdk \
 && chown -R ${UID}:${GID} /workspace

# 3) Python 의존성 설치 (공용)
#    qubee 컴파일러는 ONNX 를 입력으로 받으므로 onnx 를 포함한다.
#    QB compile smoke 에는 CUDA wheel 이 필요 없으므로 torch/torchvision 은 CPU wheel 로 고정한다.
COPY --chown=${UID}:${GID} requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        --index-url ${PYTORCH_INDEX_URL} \
        torch torchvision \
    && pip install --no-cache-dir \
        -r /tmp/requirements.txt

# 4) Mobilint SDK 설치 (vendor-provided)
#    - qubee   : ONNX -> .mxq 양자화 컴파일러
#    - qbruntime: QB-RUNTIME (.mxq 추론) + mobilint-cli
#    공개 PyPI 에 없으므로, 벤더에게 받은 wheel 을 vendor/ 에 넣어두면 빌드시 설치한다.
#    (설치 방법/패키지 명은 docs.mobilint.com 참조. maccel 이 아니라 qbruntime 을 사용한다.)
COPY --chown=${UID}:${GID} vendor/ /tmp/vendor/
RUN --mount=type=cache,target=/root/.cache/pip \
    if ls /tmp/vendor/*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /tmp/vendor/*.whl ; \
    else \
        echo "[WARN] no Mobilint wheels under vendor/. qubee/qbruntime NOT installed in image." ; \
        echo "       Place vendor-provided qubee + qbruntime wheels in vendor/ and rebuild." ; \
    fi

# 5) unified-sdk 소스 복사 및 설치
COPY --chown=${UID}:${GID} . /workspace/unified-sdk
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir . \
    && rm -f /tmp/requirements.txt

CMD ["/bin/bash"]
