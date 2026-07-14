# syntax=docker/dockerfile:1.7
# =========================
# unified-sdk (QB / Mobilint ARISE base)
# =========================
ARG BASE_IMAGE=mobilint/qbcompiler:1.2-cpu-ubuntu22.04
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG QB_RUNTIME_PIP_SPEC=mobilint-qb-runtime
ARG MBLT_MODEL_ZOO_PIP_SPEC=mblt-model-zoo

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_INPUT=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/workspace/unified-sdk/src

# 1) 기본 패키지 및 Mobilint APT 저장소 등록
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc \
    && chmod a+r /etc/apt/keyrings/mblt.asc \
    && printf "%s\n" \
        "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse" \
        > /etc/apt/sources.list.d/mobilint.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        mobilint-cli \
        git \
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

# 4) Mobilint SDK 설치
#    - qb compiler : 공식 qbcompiler Docker 이미지를 기반으로 사용
#    - qbcompiler  : 벤더 제공 compiler wheel (Python API는 qubee 로 노출될 수 있음)
#    - qbruntime   : 공식 pip 패키지(mobilint-qb-runtime)
#    - model zoo   : 표준 fetch smoke 를 위해 mblt-model-zoo 패키지를 설치한다.
COPY --chown=${UID}:${GID} vendor/ /tmp/vendor/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "${QB_RUNTIME_PIP_SPEC}" "${MBLT_MODEL_ZOO_PIP_SPEC}" \
    && if ls /tmp/vendor/qbcompiler-*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /tmp/vendor/qbcompiler-*.whl ; \
    elif ls /tmp/vendor/qubee-*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /tmp/vendor/qubee-*.whl ; \
    else \
        echo "[WARN] no qb compiler wheel under vendor/. compiler Python API NOT installed in image." ; \
        echo "       Place a vendor-provided qbcompiler-*.whl in vendor/ and rebuild." ; \
    fi

# 5) unified-sdk 소스 복사 및 설치
COPY --chown=${UID}:${GID} . /workspace/unified-sdk
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir . \
    && rm -f /tmp/requirements.txt

CMD ["/bin/bash"]
