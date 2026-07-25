# syntax=docker/dockerfile:1.7
# =========================
# unified-sdk (NVIDIA TensorRT base)
# =========================
# tensorrt 는 베이스 이미지에 포함되어 있다. GPU 는 런타임에 --gpus all 로 전달한다.
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:24.03-py3
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000
ARG VIDEO_GID=44
ARG RENDER_GID=110
# TensorRT 엔진 빌드에는 CUDA torch 가 필요 없다 (ONNX 내보내기 용도).
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.2.2
ARG TORCHVISION_VERSION=0.17.2
ARG TRT_LLM_VERSION=0.10.0

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_INPUT=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PYTHONPATH=/workspace/unified-sdk/src

# 1) 사용자/런타임 그룹 생성
RUN groupadd -g ${VIDEO_GID} video 2>/dev/null || true \
 && groupadd -g ${RENDER_GID} render 2>/dev/null || true \
 && groupadd -g ${GID} ${USERNAME} 2>/dev/null || true \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} 2>/dev/null || true

WORKDIR /workspace/unified-sdk
RUN mkdir -p /workspace/unified-sdk \
 && chown -R ${UID}:${GID} /workspace

# 2) Python 의존성
COPY --chown=${UID}:${GID} requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 8 --timeout 300 \
        --index-url ${PYTORCH_INDEX_URL} \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    && pip install --retries 8 --timeout 300 -r /tmp/requirements.txt \
    && pip install --retries 8 --timeout 300 \
         --extra-index-url https://pypi.nvidia.com \
         tensorrt_llm==${TRT_LLM_VERSION}

# 3) unified-sdk 소스 복사 및 설치
COPY --chown=${UID}:${GID} . /workspace/unified-sdk
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 8 --timeout 300 . \
    && rm -f /tmp/requirements.txt

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
