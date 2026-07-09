#!/bin/bash
set -e

# =====================================
# unified-sdk (FuriosaAI Warboy) build script
# =====================================

IMAGE_NAME="unified-sdk"
TAG="warboy"
CONTAINER_NAME=""
WORKSPACE_DIR=""
BASE_IMAGE="${WARBOY_BASE_IMAGE:-ubuntu:22.04}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
FURIOSA_SDK_VERSION="${FURIOSA_SDK_VERSION:-0.10.2}"
FURIOSA_PIP_INDEX="${FURIOSA_PIP_INDEX:-}"
WARBOY_DEVICE="${WARBOY_DEVICE:-}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_DEVICE_ARGS=()
DOCKER_TOOL_MOUNTS=()

print_usage() {
  echo "Usage: $0 [--name <container_name>] [--workspace <repo_path>] [--base-image <image>] [--pytorch-index-url <url>] [--furiosa-sdk-version <ver>] [--furiosa-pip-index <url>] [--device <node>]"
  echo ""
  echo "Options:"
  echo "  --name        Container name (default: ${IMAGE_NAME}_${TAG}_dev)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  --base-image  Docker base image used for build (default: ${BASE_IMAGE})"
  echo "  --pytorch-index-url  PyTorch wheel index (default: ${PYTORCH_INDEX_URL})"
  echo "  --furiosa-sdk-version  furiosa-sdk version (default: ${FURIOSA_SDK_VERSION})"
  echo "  --furiosa-pip-index    optional extra pip index for furiosa packages"
  echo "  --device      Warboy device node, e.g. /dev/npu0"
  echo "                (default: auto-detect /dev/npu*)"
  echo "  -h, --help    Show this help message"
}

detect_runtime_mounts() {
  if [ -n "${WARBOY_DEVICE}" ]; then
    DOCKER_DEVICE_ARGS+=( "--device" "${WARBOY_DEVICE}:${WARBOY_DEVICE}" )
    return
  fi

  for dev in /dev/npu*; do
    if [ -c "${dev}" ]; then
      DOCKER_DEVICE_ARGS+=( "--device" "${dev}:${dev}" )
    fi
  done

  TOOL_CANDIDATES=(
    "$(command -v furiosactl 2>/dev/null || true)"
    "$(command -v furiosa-smi 2>/dev/null || true)"
    /usr/bin/furiosactl
    /usr/bin/furiosa-smi
  )

  for tool in "${TOOL_CANDIDATES[@]}"; do
    if [ -f "${tool}" ]; then
      case " ${DOCKER_TOOL_MOUNTS[*]} " in
        *" ${tool}:${tool} "*) ;;
        *) DOCKER_TOOL_MOUNTS+=( "-v" "${tool}:${tool}" ) ;;
      esac
    fi
  done
}

print_run_hint() {
  echo "docker run -it --security-opt seccomp=unconfined \\"
  echo "  --name ${CONTAINER_NAME} \\"
  for ((i=0; i<${#DOCKER_DEVICE_ARGS[@]}; i+=2)); do
    echo "  ${DOCKER_DEVICE_ARGS[i]} ${DOCKER_DEVICE_ARGS[i+1]} \\"
  done
  echo "  -w /workspace/unified-sdk \\"
  echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
  for ((i=0; i<${#DOCKER_TOOL_MOUNTS[@]}; i+=2)); do
    echo "  ${DOCKER_TOOL_MOUNTS[i]} ${DOCKER_TOOL_MOUNTS[i+1]} \\"
  done
  echo "  ${IMAGE_NAME}:${TAG}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [ -z "$2" ] && { echo "[ERROR] --name requires a value"; exit 1; }
      CONTAINER_NAME="$2"; shift 2 ;;
    --workspace)
      [ -z "$2" ] && { echo "[ERROR] --workspace requires a value"; exit 1; }
      WORKSPACE_DIR="$2"; shift 2 ;;
    --base-image)
      [ -z "$2" ] && { echo "[ERROR] --base-image requires a value"; exit 1; }
      BASE_IMAGE="$2"; shift 2 ;;
    --pytorch-index-url)
      [ -z "$2" ] && { echo "[ERROR] --pytorch-index-url requires a value"; exit 1; }
      PYTORCH_INDEX_URL="$2"; shift 2 ;;
    --furiosa-sdk-version)
      [ -z "$2" ] && { echo "[ERROR] --furiosa-sdk-version requires a value"; exit 1; }
      FURIOSA_SDK_VERSION="$2"; shift 2 ;;
    --furiosa-pip-index)
      [ -z "$2" ] && { echo "[ERROR] --furiosa-pip-index requires a value"; exit 1; }
      FURIOSA_PIP_INDEX="$2"; shift 2 ;;
    --device)
      [ -z "$2" ] && { echo "[ERROR] --device requires a value"; exit 1; }
      WARBOY_DEVICE="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; print_usage; exit 1 ;;
  esac
done

[ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="${IMAGE_NAME}_${TAG}_dev"
[ -z "${WORKSPACE_DIR}" ] && WORKSPACE_DIR="${PROJECT_ROOT}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
  echo "[ERROR] Workspace directory not found: ${WORKSPACE_DIR}"
  exit 1
fi
WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd)"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile     : ${PROJECT_ROOT}/Dockerfile"
echo "  Container name : ${CONTAINER_NAME}"
echo "  Workspace(repo): ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  PyTorch index  : ${PYTORCH_INDEX_URL}"
echo "  Furiosa SDK    : ${FURIOSA_SDK_VERSION}"
echo "  Furiosa index  : ${FURIOSA_PIP_INDEX:-public PyPI}"
echo "  Device         : ${WARBOY_DEVICE:-auto}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL}" \
  --build-arg FURIOSA_SDK_VERSION="${FURIOSA_SDK_VERSION}" \
  --build-arg FURIOSA_PIP_INDEX="${FURIOSA_PIP_INDEX}" \
  .

detect_runtime_mounts

echo "Build complete!"
echo ""
if [ ${#DOCKER_DEVICE_ARGS[@]} -eq 0 ]; then
  echo "[WARN] No Warboy device nodes were detected on this host."
  echo "       Expected at least one /dev/npu* character device (host driver required)."
  echo "       Pass one explicitly with --device /dev/npu0 if needed."
  echo ""
fi

echo "Run container with:"
print_run_hint

echo ""
echo "Sanity check inside container:"
echo "  furiosactl list && furiosactl info || true"
echo "  furiosa-compiler --version || true"
echo "  python3 -c \"import unified_sdk; from furiosa.runtime import sync; print('OK')\""
echo "  python3 examples/run_warboy_build.py --help"
