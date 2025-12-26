# =========================
# unified-sdk (TensorRT base)
# =========================
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:24.03-py3
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# 기본 작업 디렉토리
WORKDIR /workspace
RUN chown ${UID}:${GID} /workspace

# 1) requirements 설치
COPY --chown=${UID}:${GID} requirements.txt /workspace/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=secret,id=netrc,target=/root/.netrc,mode=0600 \
    pip3 install --no-cache-dir -r requirements.txt

# 2) unified-sdk 소스 복사
#   → 호스트에서 Docker build를 할 때, 현재 디렉토리(.) 안에 unified-sdk 폴더가 있어야 함
COPY --chown=${UID}:${GID} . /workspace/unified-sdk

# 3) unified-sdk 패키지 editable 설치
WORKDIR /workspace/unified-sdk
# RUN pip install --no-cache-dir -e .
RUN --mount=type=secret,id=netrc,target=/root/.netrc,mode=0600 \
    pip install --no-cache-dir -e .

# 4) workspace로 다시 돌아오기
WORKDIR /workspace

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
