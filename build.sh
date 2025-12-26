#!/bin/bash
set -e


# =============================
# unified-sdk build script
# =============================

IMAGE_NAME="unified-sdk"
TAG="cuda12.3"
BASE_IMAGE="nvcr.io/nvidia/tensorrt:23.10-py3"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

echo " Building Docker image: ${IMAGE_NAME}:${TAG}"
echo " Base image: ${BASE_IMAGE}"
echo " UID:GID = ${UID_VALUE}:${GID_VALUE}"

# Build (no cache option if needed)
DOCKER_BUILDKIT=1 docker build \
  --secret id=netrc,src=.secrets/netrc \
  -t ${IMAGE_NAME}:${TAG} \
  --build-arg BASE_IMAGE=${BASE_IMAGE} \
  --build-arg UID=${UID_VALUE} \
  --build-arg GID=${GID_VALUE} \
  .

echo " Build complete!"


# echo " Run container with:"
# echo " docker run --gpus all -it --security-opt seccomp=unconfined --name ${IMAGE_NAME}_dev -v /home/etri/users/rskim/uDC:/workspace ${IMAGE_NAME}:${TAG}"


########################################
# NVIDIA Docker 모드 자동 감지
########################################

detect_nvidia_mode() {
  # 1) --gpus all 테스트
  if docker run --rm --gpus all hello-world >/dev/null 2>&1; then
    echo "gpus"
    return 0
  fi

  # 2) --runtime=nvidia 테스트
  if docker run --rm --runtime=nvidia hello-world >/dev/null 2>&1; then
    echo "runtime"
    return 0
  fi

  # 3) 둘 다 안 되면
  echo "none"
  return 0
}

MODE=$(detect_nvidia_mode)

echo " Detected NVIDIA Docker mode: ${MODE}"

case "${MODE}" in
  gpus)
    GPU_FLAG="--gpus all"
    ;;
  runtime)
    GPU_FLAG="--runtime=nvidia"
    ;;
  none)
    GPU_FLAG=""
    ;;
esac


########################################
# 실행 예시 출력
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "${MODE}" = "none" ]; then
  echo " [WARN] --gpus all / --runtime=nvidia 둘 다 동작하지 않습니다."
  echo "        GPU 설정이 다른 환경일 수 있으니, 필요시 직접 옵션을 추가하세요."
  echo ""
  echo " Run container (WITHOUT explicit GPU option) with:"
  echo " docker run -it --security-opt seccomp=unconfined \\"
  echo "   --name ${IMAGE_NAME}_dev \\"
  echo "   -v /path/to/parent_folder:/workspace \\"
  echo "   ${IMAGE_NAME}:${TAG} \\"
  echo "   # e.g. \\"
  echo "   docker run -it --security-opt seccomp=unconfined \\"
  echo "   --name ${IMAGE_NAME}_dev \\"
  echo "   -v ${PARENT_DIR}:/workspace"
  echo "   ${IMAGE_NAME}:${TAG}"
  echo ""
  echo " Run container (WITH RB/ARIES devices) with:"
  echo " docker run ${GPU_FLAG} -it --security-opt seccomp=unconfined \\"
  echo "   --name ${IMAGE_NAME}_dev \\"
  echo "   --device /dev/rsd0:/dev/rsd0 \\"
  echo "   --device /dev/rbln0:/dev/rbln0 \\"
  echo "   --device /dev/rbln1:/dev/rbln1 \\"
  echo "   --device /dev/rbln2:/dev/rbln2 \\"
  echo "   --device /dev/aries0:/dev/aries0 \\"
  echo "   -v ${PARENT_DIR}:/workspace \\"
  echo "   -v /usr/local/bin/rbln-smi:/usr/local/bin/rbln-smi \\"
  echo "   -v /usr/local/bin/rbln-stat:/usr/local/bin/rbln-stat \\"
  echo "   ${IMAGE_NAME}:${TAG}"

else
  echo " Run container with:"
  echo " docker run ${GPU_FLAG} -it --security-opt seccomp=unconfined \\"
  echo "   --name ${IMAGE_NAME}_dev \\"
  echo "   -v /path/to/parent_folder:/workspace \\"
  echo "   ${IMAGE_NAME}:${TAG}"
  echo "   # e.g.\\"
  echo " docker run ${GPU_FLAG} -it --security-opt seccomp=unconfined \\"
  echo "   --name ${IMAGE_NAME}_dev \\"
  echo "   -v ${PARENT_DIR}:/workspace \\"
  echo "   ${IMAGE_NAME}:${TAG}"
  echo ""
  echo " Run container (WITH RB/ARIES devices) with:"
  echo " docker run ${GPU_FLAG} -it --security-opt seccomp=unconfined \\"
  echo "   --name ${IMAGE_NAME}_dev \\"
  echo "   --device /dev/rsd0:/dev/rsd0 \\"
  echo "   --device /dev/rbln0:/dev/rbln0 \\"
  echo "   --device /dev/rbln1:/dev/rbln1 \\"
  echo "   --device /dev/rbln2:/dev/rbln2 \\"
  echo "   --device /dev/aries0:/dev/aries0 \\"
  echo "   -v ${PARENT_DIR}:/workspace \\"
  echo "   -v /usr/local/bin/rbln-smi:/usr/local/bin/rbln-smi \\"
  echo "   -v /usr/local/bin/rbln-stat:/usr/local/bin/rbln-stat \\"
  echo "   ${IMAGE_NAME}:${TAG}"
  
fi
