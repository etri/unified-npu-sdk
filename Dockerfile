# syntax=docker/dockerfile:1.7
# =========================
# unified-sdk (RBLN base)
# =========================
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000
ARG REBEL_COMPILER_VERSION=0.11.0
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
#    RBLN compile smoke에는 CUDA wheel이 필요 없으므로 torch/torchvision은 CPU wheel로 고정 설치한다.
COPY --chown=${UID}:${GID} requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        --index-url ${PYTORCH_INDEX_URL} \
        torch torchvision \
    && pip install --no-cache-dir \
        -r /tmp/requirements.txt

# 4) rebel-compiler(Rebellions SDK) 설치
#    - pypi.rbln.ai 인증을 위해 BuildKit secret(.secrets/netrc) 마운트
#    - 공식 가이드 형식: PyPI primary + rbln extra
#    - 현재 기본값은 RBLN SDK 0.11.0 검증 기준에 맞춤
RUN --mount=type=secret,id=netrc,target=/root/.netrc,mode=0600 \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        --extra-index-url https://pypi.rbln.ai/simple \
        rebel-compiler==${REBEL_COMPILER_VERSION}

# 5) unified-sdk 소스 복사 및 설치
COPY --chown=${UID}:${GID} . /workspace/unified-sdk
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir . \
    && rm /tmp/requirements.txt

CMD ["/bin/bash"]
