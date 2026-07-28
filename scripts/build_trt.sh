#!/bin/bash
set -e

IMAGE_NAME="unified-sdk"
FLAVOR="vision"
TAG=""
CONTAINER_NAME=""
WORKSPACE_DIR=""
VISION_BASE_IMAGE="${TRT_VISION_BASE_IMAGE:-nvcr.io/nvidia/tensorrt:24.03-py3}"
LLM_BASE_IMAGE="${TRT_LLM_BASE_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22}"
BASE_IMAGE=""
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)
RENDER_GID_VALUE="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID_VALUE="$(getent group video | cut -d: -f3 || true)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/Dockers"
GPU_FLAG=""

print_usage() {
  echo "사용법: $0 [--flavor vision|llm] [-n <container_name>] [--workspace <repo_path>] [--base-image <image>]"
  echo ""
  echo "옵션:"
  echo "  --flavor      Docker flavor 선택 (기본: vision)"
  echo "  -n, --name    컨테이너 이름 (기본: trt-only-<flavor>)"
  echo "  --workspace   /workspace/unified-sdk 로 마운트할 호스트 repo 경로 (기본: 현재 프로젝트 루트)"
  echo "  --base-image  빌드에 사용할 Docker base image (기본: flavor별 권장 이미지)"
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
  if [ "${FLAVOR}" = "llm" ]; then
    echo "  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\"
  fi
  echo "  -w /workspace/unified-sdk \\"
  echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
  echo "  ${IMAGE_NAME}:${TAG}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor)
      [ -z "$2" ] && { echo "[ERROR] --flavor 값이 필요합니다"; exit 1; }
      FLAVOR="$2"; shift 2 ;;
    -n|--name)
      [ -z "$2" ] && { echo "[ERROR] --name 값이 필요합니다"; exit 1; }
      CONTAINER_NAME="$2"; shift 2 ;;
    --workspace)
      [ -z "$2" ] && { echo "[ERROR] --workspace 값이 필요합니다"; exit 1; }
      WORKSPACE_DIR="$2"; shift 2 ;;
    --base-image)
      [ -z "$2" ] && { echo "[ERROR] --base-image 값이 필요합니다"; exit 1; }
      BASE_IMAGE="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "[ERROR] 알 수 없는 옵션: $1"; print_usage; exit 1 ;;
  esac
done

case "${FLAVOR}" in
  vision)
    TAG="main-trt-vision"
    [ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="main-trt-vision"
    [ -z "${BASE_IMAGE}" ] && BASE_IMAGE="${VISION_BASE_IMAGE}"
    DOCKERFILE_PATH="${DOCKER_DIR}/docker.trt.vision"
    ;;
  llm)
    TAG="main-trt-llm"
    [ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="main-trt-llm"
    [ -z "${BASE_IMAGE}" ] && BASE_IMAGE="${LLM_BASE_IMAGE}"
    DOCKERFILE_PATH="${DOCKER_DIR}/docker.trt.llm"
    ;;
  *)
    echo "[ERROR] --flavor 는 vision 또는 llm 이어야 합니다: ${FLAVOR}"
    exit 1
    ;;
esac

[ -z "${WORKSPACE_DIR}" ] && WORKSPACE_DIR="${PROJECT_ROOT}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
  echo "[ERROR] 워크스페이스 디렉터리를 찾을 수 없습니다: ${WORKSPACE_DIR}"
  exit 1
fi
WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd)"

echo "Docker 이미지 빌드: ${IMAGE_NAME}:${TAG}"
echo "  Flavor         : ${FLAVOR}"
echo "  Dockerfile     : ${DOCKERFILE_PATH}"
echo "  컨테이너 이름  : ${CONTAINER_NAME}"
echo "  워크스페이스   : ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"
if [ -n "${VIDEO_GID_VALUE}" ]; then
  echo "  Video GID      : ${VIDEO_GID_VALUE}"
fi
if [ -n "${RENDER_GID_VALUE}" ]; then
  echo "  Render GID     : ${RENDER_GID_VALUE}"
fi

cd "${PROJECT_ROOT}"

if [ "${FLAVOR}" = "vision" ]; then
  DOCKER_BUILDKIT=1 docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}:${TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg UID="${UID_VALUE}" \
    --build-arg GID="${GID_VALUE}" \
    --build-arg VIDEO_GID="${VIDEO_GID_VALUE:-44}" \
    --build-arg RENDER_GID="${RENDER_GID_VALUE:-110}" \
    .
else
  DOCKER_BUILDKIT=1 docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}:${TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg UID="${UID_VALUE}" \
    --build-arg GID="${GID_VALUE}" \
    --build-arg VIDEO_GID="${VIDEO_GID_VALUE:-44}" \
    --build-arg RENDER_GID="${RENDER_GID_VALUE:-110}" \
    .
fi

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
if [ "${FLAVOR}" = "vision" ]; then
  echo "  python3 -c \"import tensorrt as trt; from importlib import metadata; print('tensorrt=', getattr(trt, '__version__', metadata.version('tensorrt')))\""
  echo "  python3 examples/run_tensorrt_build.py --help"
  echo "  python3 examples/run_tensorrt_infer.py --help"
else
  echo "  python3 -c \"import tensorrt_llm; print('tensorrt_llm OK')\""
  echo "  python3 examples/run_tensorrt_llm_build.py --help"
  echo "  python3 examples/run_tensorrt_llm_infer.py --help"
  echo "  python3 examples/inspect_tensorrt_llm_model.py --help"
fi
