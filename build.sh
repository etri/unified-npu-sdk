#!/bin/bash
set -e

# =====================================
# unified-sdk (RBLN) build script
# =====================================

IMAGE_NAME="unified-sdk"
TAG="rbln"
CONTAINER_NAME=""
WORKSPACE_DIR=""
BASE_IMAGE="${RBLN_BASE_IMAGE:-ubuntu:22.04}"
COMPILER_VERSION="${REBEL_COMPILER_VERSION:-0.11.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
CDI_DEVICE="${RBLN_CDI_DEVICE:-}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_DEVICE_ARGS=()
DOCKER_TOOL_MOUNTS=()

print_usage() {
  echo "Usage: $0 [--name <container_name>] [--workspace <repo_path>] [--base-image <image>] [--compiler-version <version>] [--pytorch-index-url <url>]"
  echo ""
  echo "Options:"
  echo "  --name        Container name (default: ${IMAGE_NAME}_${TAG}_dev)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  --base-image  Docker base image used for build"
  echo "                (default: ${BASE_IMAGE})"
  echo "  --compiler-version  rebel-compiler version to install during docker build"
  echo "                (default: ${COMPILER_VERSION})"
  echo "  --pytorch-index-url  PyTorch wheel index used for torch/torchvision"
  echo "                (default: ${PYTORCH_INDEX_URL})"
  echo "  --cdi-device  RBLN CDI device handle, e.g. rebellions.ai/npu=all"
  echo "                (default: auto-detect /var/run/cdi/rbln.yaml, then manual /dev/rbln*)"
  echo "  -h, --help    Show this help message"
}

detect_runtime_mounts() {
  if [ -n "${CDI_DEVICE}" ]; then
    DOCKER_DEVICE_ARGS+=( "--device" "${CDI_DEVICE}" )
    return
  fi

  if [ -f /var/run/cdi/rbln.yaml ]; then
    CDI_DEVICE="rebellions.ai/npu=all"
    DOCKER_DEVICE_ARGS+=( "--device" "${CDI_DEVICE}" )
    return
  fi

  for dev in /dev/rbln*; do
    if [ -c "${dev}" ]; then
      DOCKER_DEVICE_ARGS+=( "--device" "${dev}:${dev}" )
    fi
  done

  TOOL_CANDIDATES=(
    "$(command -v rbln-smi 2>/dev/null || true)"
    "$(command -v rbln-stat 2>/dev/null || true)"
    /usr/local/bin/rbln-smi
    /usr/local/bin/rbln-stat
    /usr/bin/rbln-smi
    /usr/bin/rbln-stat
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
    --compiler-version)
      [ -z "$2" ] && { echo "[ERROR] --compiler-version requires a value"; exit 1; }
      COMPILER_VERSION="$2"; shift 2 ;;
    --pytorch-index-url)
      [ -z "$2" ] && { echo "[ERROR] --pytorch-index-url requires a value"; exit 1; }
      PYTORCH_INDEX_URL="$2"; shift 2 ;;
    --cdi-device)
      [ -z "$2" ] && { echo "[ERROR] --cdi-device requires a value"; exit 1; }
      CDI_DEVICE="$2"; shift 2 ;;
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

# Rebellions SDK 인증 파일 (.secrets/netrc) - 필수
NETRC_PATH="${PROJECT_ROOT}/.secrets/netrc"
if [ ! -f "${NETRC_PATH}" ]; then
  echo "[ERROR] Rebellions credential file not found: ${NETRC_PATH}"
  echo ""
  echo "Create it first:"
  echo "  mkdir -p ${PROJECT_ROOT}/.secrets"
  echo "  cat > ${NETRC_PATH} <<'EOF'"
  echo "  machine pypi.rbln.ai"
  echo "  login YOUR_RBLN_USERNAME"
  echo "  password YOUR_RBLN_PASSWORD"
  echo "  EOF"
  exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile     : ${PROJECT_ROOT}/Dockerfile"
echo "  Container name : ${CONTAINER_NAME}"
echo "  Workspace(repo): ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  Compiler ver.  : ${COMPILER_VERSION}"
echo "  PyTorch index  : ${PYTORCH_INDEX_URL}"
echo "  CDI device     : ${CDI_DEVICE:-auto}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  --secret "id=netrc,src=${NETRC_PATH}" \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg REBEL_COMPILER_VERSION="${COMPILER_VERSION}" \
  --build-arg PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL}" \
  .

detect_runtime_mounts

echo "Build complete!"
echo ""
if [ ${#DOCKER_DEVICE_ARGS[@]} -eq 0 ]; then
  echo "[WARN] No RBLN device nodes were detected on this host."
  echo "       Preferred: configure RBLN Container Toolkit CDI and use rebellions.ai/npu=all."
  echo "       Fallback expected at least one /dev/rbln* character device."
  echo ""
elif [ -n "${CDI_DEVICE}" ]; then
  echo "[INFO] Using RBLN CDI device handle: ${CDI_DEVICE}"
  echo ""
fi

echo "Run container with:"
print_run_hint

echo ""
echo "Sanity check inside container:"
echo "  command -v rbln-smi && rbln-smi || true"
echo "  python3 -c \"import unified_sdk, rebel; print('OK')\""
echo "  python3 -c \"import rebel; print('npu_is_available=', rebel.npu_is_available())\""
echo "  python3 -c \"import torch, torchvision, rebel; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('rebel=', getattr(rebel, '__version__', 'unknown'))\""
echo "  RBLN_DEVICES=0 python3 examples/run_rbln_build.py"
