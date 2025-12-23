# =========================
# unified-sdk (TensorRT base)
# =========================
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:24.03-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /workspace

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}
RUN mkdir -p /workspace && chown -R ${UID}:${GID} /workspace

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]

