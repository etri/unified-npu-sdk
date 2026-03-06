#!/bin/bash
set -e

IMAGE_NAME="unified-sdk"
TARGET="tensorrt"
CONTAINER_NAME=""
WORKSPACE_DIR=""
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

print_usage() {
  echo "Usage: $0 [--target <target>] [--name <container_name>] [--workspace <repo_path>]"
  echo ""
  echo "Options:"
  echo "  --target      Build target (default: tensorrt)"
  echo "                Supported: tensorrt, rebellions, furiosa"
  echo "  --name        Container name (default: unified-sdk_<target>_dev)"
  echo "  --workspace   Host repository path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  -h, --help    Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0"
  echo "  $0 --target tensorrt"
  echo "  $0 --target tensorrt --name rskim_unified-npu-sdk"
  echo "  $0 --target tensorrt --workspace /home/etri/users/rskim/uDC/unified-npu-sdk"
  echo "  $0 --target tensorrt --name rskim_unified-npu-sdk --workspace /home/etri/users/rskim/uDC/unified-npu-sdk"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [ -z "$2" ]; then
        echo "[ERROR] --target requires a value"
        exit 1
      fi
      TARGET="$2"
      shift 2
      ;;
    --name)
      if [ -z "$2" ]; then
        echo "[ERROR] --name requires a value"
        exit 1
      fi
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --workspace)
      if [ -z "$2" ]; then
        echo "[ERROR] --workspace requires a value"
        exit 1
      fi
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      echo ""
      print_usage
      exit 1
      ;;
  esac
done

if [ -z "${CONTAINER_NAME}" ]; then
  CONTAINER_NAME="${IMAGE_NAME}_${TARGET}_dev"
fi

if [ -z "${WORKSPACE_DIR}" ]; then
  WORKSPACE_DIR="${PROJECT_ROOT}"
fi

if [ ! -d "${WORKSPACE_DIR}" ]; then
  echo "[ERROR] Workspace directory not found: ${WORKSPACE_DIR}"
  exit 1
fi

WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd)"

SECRET_ARGS=()
if [ -f "${PROJECT_ROOT}/.secrets/netrc" ]; then
  SECRET_ARGS=(--secret "id=netrc,src=${PROJECT_ROOT}/.secrets/netrc")
fi

case "${TARGET}" in
  tensorrt)
    TAG="tensorrt"
    BASE_IMAGE="nvcr.io/nvidia/tensorrt:23.10-py3"
    DOCKERFILE_PATH="${PROJECT_ROOT}/Dockerfiles/tensorrt/Dockerfile"
    NEED_GPU="yes"
    ;;
  rebellions)
    TAG="rebellions"
    BASE_IMAGE="ubuntu:24.04"
    DOCKERFILE_PATH="${PROJECT_ROOT}/Dockerfiles/rebellions/Dockerfile"
    NEED_GPU="no"
    ;;
  furiosa)
    TAG="furiosa"
    BASE_IMAGE="ubuntu:24.04"
    DOCKERFILE_PATH="${PROJECT_ROOT}/Dockerfiles/furiosa/Dockerfile"
    NEED_GPU="no"
    ;;
  *)
    echo "[ERROR] Unsupported target: ${TARGET}"
    echo "Supported targets: tensorrt, rebellions, furiosa"
    exit 1
    ;;
esac

if [ ! -f "${DOCKERFILE_PATH}" ]; then
  echo "[ERROR] Dockerfile not found: ${DOCKERFILE_PATH}"
  exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Target: ${TARGET}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "Base image: ${BASE_IMAGE}"
echo "Container name: ${CONTAINER_NAME}"
echo "Workspace(repo): ${WORKSPACE_DIR}"
echo "UID:GID = ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  "${SECRET_ARGS[@]}" \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  .

echo "Build complete!"

########################################
# NVIDIA Docker mode auto detect
########################################

detect_nvidia_mode() {
  if docker run --rm --gpus all hello-world >/dev/null 2>&1; then
    echo "gpus"
    return 0
  fi

  if docker run --rm --runtime=nvidia hello-world >/dev/null 2>&1; then
    echo "runtime"
    return 0
  fi

  echo "none"
  return 0
}

MODE=$(detect_nvidia_mode)

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

echo ""
echo "Detected NVIDIA Docker mode: ${MODE}"
echo ""

########################################
# Run command output
########################################

if [ "${NEED_GPU}" = "yes" ]; then
  if [ "${MODE}" = "none" ]; then
    echo "[WARN] GPU runtime was not detected automatically."
    echo "       Please check NVIDIA Container Toolkit installation."
    echo ""
    echo "Run container with:"
    echo "docker run -it --security-opt seccomp=unconfined \\"
    echo "  --name ${CONTAINER_NAME} \\"
    echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
    echo "  ${IMAGE_NAME}:${TAG}"
  else
    echo "Run container with:"
    echo "docker run ${GPU_FLAG} -it --security-opt seccomp=unconfined \\"
    echo "  --name ${CONTAINER_NAME} \\"
    echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
    echo "  ${IMAGE_NAME}:${TAG}"
  fi
else
  echo "Run container with:"
  echo "docker run -it --security-opt seccomp=unconfined \\"
  echo "  --name ${CONTAINER_NAME} \\"
  echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
  echo "  ${IMAGE_NAME}:${TAG}"
fi

echo ""
echo "After entering the container, test with:"
echo "  python -m pip show unified-sdk"
echo "  python -c \"import unified_sdk; print('unified_sdk import ok')\""
