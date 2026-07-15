#!/bin/bash
set -e

# =====================================
# unified-sdk (NVIDIA TensorRT) 빌드 스크립트
# =====================================

IMAGE_NAME="unified-sdk"
TAG="tensorrt"
CONTAINER_NAME=""
WORKSPACE_DIR=""
BASE_IMAGE="${TRT_BASE_IMAGE:-nvcr.io/nvidia/tensorrt:24.03-py3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)
RENDER_GID_VALUE="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID_VALUE="$(getent group video | cut -d: -f3 || true)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

GPU_FLAG=""

print_usage() {
  echo "사용법: $0 [-n <container_name>] [--workspace <repo_path>] [--base-image <image>] [--pytorch-index-url <url>]"
  echo ""
  echo "옵션:"
  echo "  -n, --name    컨테이너 이름 (기본: tensorrt-only)"
  echo "  --workspace   /workspace/unified-sdk 로 마운트할 호스트 repo 경로 (기본: 현재 프로젝트 루트)"
  echo "  --base-image  빌드에 사용할 Docker base image (기본: ${BASE_IMAGE})"
  echo "  --pytorch-index-url  torch/torchvision wheel 인덱스 (기본: ${PYTORCH_INDEX_URL})"
  echo "  -h, --help    도움말 출력"
}

detect_nvidia_mode() {
  if docker run --rm --gpus all hello-world >/dev/null 2>&1; then
    echo "gpus"; return 0
  fi
  if docker run --rm --runtime=nvidia hello-world >/dev/null 2>&1; then
    echo "runtime"; return 0
  fi
  echo "none"; return 0
}

print_run_hint() {
  echo "docker run ${GPU_FLAG} -it --security-opt seccomp=unconfined \\"
  echo "  --name ${CONTAINER_NAME} \\"
  echo "  -w /workspace/unified-sdk \\"
  echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
  echo "  ${IMAGE_NAME}:${TAG}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      [ -z "$2" ] && { echo "[ERROR] --name 값이 필요합니다"; exit 1; }
      CONTAINER_NAME="$2"; shift 2 ;;
    --workspace)
      [ -z "$2" ] && { echo "[ERROR] --workspace 값이 필요합니다"; exit 1; }
      WORKSPACE_DIR="$2"; shift 2 ;;
    --base-image)
      [ -z "$2" ] && { echo "[ERROR] --base-image 값이 필요합니다"; exit 1; }
      BASE_IMAGE="$2"; shift 2 ;;
    --pytorch-index-url)
      [ -z "$2" ] && { echo "[ERROR] --pytorch-index-url 값이 필요합니다"; exit 1; }
      PYTORCH_INDEX_URL="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "[ERROR] 알 수 없는 옵션: $1"; print_usage; exit 1 ;;
  esac
done

[ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="tensorrt-only"
[ -z "${WORKSPACE_DIR}" ] && WORKSPACE_DIR="${PROJECT_ROOT}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
  echo "[ERROR] 워크스페이스 디렉터리를 찾을 수 없습니다: ${WORKSPACE_DIR}"
  exit 1
fi
WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd)"

echo "Docker 이미지 빌드: ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile     : ${PROJECT_ROOT}/Dockerfile"
echo "  컨테이너 이름  : ${CONTAINER_NAME}"
echo "  워크스페이스   : ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  PyTorch index  : ${PYTORCH_INDEX_URL}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"
if [ -n "${VIDEO_GID_VALUE}" ]; then
  echo "  Video GID      : ${VIDEO_GID_VALUE}"
fi
if [ -n "${RENDER_GID_VALUE}" ]; then
  echo "  Render GID     : ${RENDER_GID_VALUE}"
fi

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg VIDEO_GID="${VIDEO_GID_VALUE:-44}" \
  --build-arg RENDER_GID="${RENDER_GID_VALUE:-110}" \
  --build-arg PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL}" \
  .

echo "빌드 완료!"
echo ""

MODE=$(detect_nvidia_mode)
echo "감지된 NVIDIA Docker 모드: ${MODE}"
case "${MODE}" in
  gpus)    GPU_FLAG="--gpus all" ;;
  runtime) GPU_FLAG="--runtime=nvidia" ;;
  none)    GPU_FLAG="" ;;
esac

if [ "${MODE}" = "none" ]; then
  echo "[WARN] --gpus all / --runtime=nvidia 둘 다 동작하지 않습니다."
  echo "       GPU 설정이 다른 환경일 수 있으니 필요 시 직접 옵션을 추가하세요."
  echo ""
fi

echo "컨테이너 실행:"
print_run_hint

echo ""
echo "컨테이너 내부 점검:"
echo "  nvidia-smi || true"
echo "  python3 -c \"import unified_sdk; print('OK')\""
echo "  python3 -c \"import tensorrt; print('tensorrt=', tensorrt.__version__)\""
echo "  python3 examples/run_tensorrt_build.py --help"
