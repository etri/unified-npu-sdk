# syntax=docker/dockerfile:1.7

# =========================
# unified-sdk (TensorRT base)
# =========================
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:23.10-py3
FROM ${BASE_IMAGE}

ARG USERNAME=etri
ARG UID=1000
ARG GID=1000

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_INPUT=1

# 컨테이너 내부 사용자 생성
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# 기본 작업 디렉토리
WORKDIR /app
RUN chown ${UID}:${GID} /app

# 1) requirements 설치
#    - requirements.txt만 먼저 복사해서 docker layer cache 효율 확보
COPY --chown=${UID}:${GID} Dockerfiles/tensorrt/requirements.txt /app/requirements.txt
RUN --mount=type=secret,id=netrc,target=/root/.netrc,mode=0600 \
    --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r /app/requirements.txt

# 2) unified-sdk 소스 복사
#    - docker build 컨텍스트는 프로젝트 루트(.)여야 함
COPY --chown=${UID}:${GID} . /app/unified-sdk

# 3) unified-sdk 패키지 editable 설치
WORKDIR /app/unified-sdk
RUN --mount=type=secret,id=netrc,target=/root/.netrc,mode=0600 \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -e . \
    && rm /app/requirements.txt

# 4) workspace로 다시 돌아오기
WORKDIR /app

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
