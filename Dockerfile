# syntax=docker/dockerfile:1.7
# =========================
# unified-sdk (FuriosaAI Warboy base)
# =========================
# Ubuntu 22.04(jammy) 기준. Warboy 는 warboy-jammy APT suite 를 사용한다.
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG FURIOSA_SDK_VERSION=0.10.2
# Furiosa pip 패키지에 별도 인덱스가 필요하면 지정 (기본은 공개 PyPI 가정).
ARG FURIOSA_PIP_INDEX=
# 컨테이너에 설치할 Warboy userspace 패키지 (커널 드라이버는 호스트 전제).
ARG FURIOSA_APT_PACKAGES="furiosa-libnux furiosa-libhal-warboy furiosa-compiler libonnxruntime"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_INPUT=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/workspace/unified-sdk/src

# 1) 기본 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git ca-certificates curl gnupg \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# 2) FuriosaAI Warboy APT suite (public: warboy-jammy) + userspace runtime/compiler
#    커널 드라이버(furiosa-driver-warboy)는 호스트에 설치되어 있어야 하며, /dev/npu* 를 컨테이너로 전달한다.
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/cloud.google.gpg \
    && echo "deb [arch=amd64] http://asia-northeast3-apt.pkg.dev/projects/furiosa-ai warboy-jammy main" \
        > /etc/apt/sources.list.d/furiosa-warboy.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${FURIOSA_APT_PACKAGES} \
    && rm -rf /var/lib/apt/lists/*

# 3) 사용자 생성
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/unified-sdk
RUN mkdir -p /workspace/unified-sdk \
 && chown -R ${UID}:${GID} /workspace

# 4) Python 의존성 (공용). Warboy compile smoke 에는 CUDA wheel 이 필요 없다.
#    furiosa-models 일부 vision model(YOLOv5/YOLOv7 계열)은 torchvision 을 사용하므로
#    torch/torchvision 을 같은 PyTorch index 에서 함께 설치해 ABI/ops mismatch 를 피한다.
COPY --chown=${UID}:${GID} requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --retries 8 --timeout 300 \
        --index-url ${PYTORCH_INDEX_URL} \
        torch torchvision \
    && pip install --no-cache-dir --retries 8 --timeout 300 \
        -r /tmp/requirements.txt

# 5) FuriosaAI SDK (Python): quantizer + runtime + Model Zoo
#    별도 pip invocation 에서 furiosa-models 의존성 resolver 가 OpenCV 후보를 다시 backtracking 하지 않도록
#    공용 requirements.txt 를 constraint 로 재사용한다.
RUN --mount=type=cache,target=/root/.cache/pip \
    EXTRA_INDEX_ARG="" ; \
    if [ -n "${FURIOSA_PIP_INDEX}" ]; then EXTRA_INDEX_ARG="--extra-index-url ${FURIOSA_PIP_INDEX}" ; fi ; \
    pip install --no-cache-dir --retries 8 --timeout 300 ${EXTRA_INDEX_ARG} -c /tmp/requirements.txt \
        "furiosa-sdk[quantizer]==${FURIOSA_SDK_VERSION}" \
        "furiosa-models==${FURIOSA_SDK_VERSION}"

# 6) unified-sdk 소스 복사 및 설치
COPY --chown=${UID}:${GID} . /workspace/unified-sdk
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --retries 8 --timeout 300 . \
    && rm -f /tmp/requirements.txt

CMD ["/bin/bash"]
