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
OPTIMUM_RBLN_VERSION="${OPTIMUM_RBLN_VERSION:-0.11.0.post1}"
VLLM_RBLN_VERSION="${VLLM_RBLN_VERSION:-0.11.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
CDI_DEVICE="${RBLN_CDI_DEVICE:-}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)
USER_MODE="${RBLN_USER_MODE:-root}"
CDI_SPEC_DETECTED=0
CDI_SPEC_HINT=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/Dockers"
DOCKERFILE_PATH="${DOCKER_DIR}/docker.rbln.unified"

DOCKER_DEVICE_ARGS=()
DOCKER_GROUP_ARGS=()

print_usage() {
  echo "Usage: $0 [-n <container_name>] [--workspace <repo_path>] [--base-image <image>] [--compiler-version <version>] [--pytorch-index-url <url>]"
  echo ""
  echo "Options:"
  echo "  -n, --name    Container name (default: rbln-only)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  --base-image  Docker base image used for build"
  echo "                (default: ${BASE_IMAGE})"
  echo "  --compiler-version  rebel-compiler version to install during docker build"
  echo "                (default: ${COMPILER_VERSION})"
  echo "  --optimum-rbln-version  optimum-rbln version to install during docker build"
  echo "                (default: ${OPTIMUM_RBLN_VERSION})"
  echo "  --vllm-rbln-version  vllm-rbln version to install during docker build"
  echo "                (default: ${VLLM_RBLN_VERSION})"
  echo "  --pytorch-index-url  PyTorch wheel index used for torch/torchvision"
  echo "                (default: ${PYTORCH_INDEX_URL})"
  echo "  --cdi-device  RBLN CDI device handle, e.g. rebellions.ai/npu=all"
  echo "                (default: auto-detect /var/run/cdi/rbln.yaml, else use rebellions.ai/npu=all)"
  echo "  --user-mode   Container user mode: root | host"
  echo "                (default: ${USER_MODE})"
  echo "  -h, --help    Show this help message"
}

detect_runtime_mounts() {
  if [ -n "${CDI_DEVICE}" ]; then
    DOCKER_DEVICE_ARGS+=( "--device" "${CDI_DEVICE}" )
    return
  fi

  for spec in /var/run/cdi/rbln.yaml /etc/cdi/rbln.yaml; do
    if [ -f "${spec}" ]; then
      CDI_SPEC_DETECTED=1
      CDI_SPEC_HINT="${spec}"
      break
    fi
  done

  if [ "${CDI_SPEC_DETECTED}" -eq 0 ] && command -v rbln-ctk >/dev/null 2>&1; then
    if rbln-ctk cdi list >/dev/null 2>&1; then
      CDI_SPEC_DETECTED=1
      CDI_SPEC_HINT="rbln-ctk cdi list"
    fi
  fi

  CDI_DEVICE="rebellions.ai/npu=all"
  DOCKER_DEVICE_ARGS+=( "--device" "${CDI_DEVICE}" )
}

detect_device_group() {
  local dev=""
  local group_id=""

  for dev in /dev/rbln0 /dev/rbln1 /dev/rebellions0 /dev/rebellions1 /dev/atom0 /dev/atom1; do
    if [ -e "${dev}" ]; then
      group_id="$(stat -c '%g' "${dev}" 2>/dev/null || true)"
      if [ -n "${group_id}" ] && [ "${group_id}" != "0" ]; then
        DOCKER_GROUP_ARGS+=( "--group-add" "${group_id}" )
        return
      fi
    fi
  done
}

detect_keep_groups_support() {
  if docker run --help 2>/dev/null | grep -q "keep-groups"; then
    DOCKER_GROUP_ARGS+=( "--group-add" "keep-groups" )
  fi
}

print_run_hint() {
  echo "docker run -it --security-opt seccomp=unconfined \\"
  echo "  --name ${CONTAINER_NAME} \\"
  for ((i=0; i<${#DOCKER_DEVICE_ARGS[@]}; i+=2)); do
    echo "  ${DOCKER_DEVICE_ARGS[i]} ${DOCKER_DEVICE_ARGS[i+1]} \\"
  done
  for ((i=0; i<${#DOCKER_GROUP_ARGS[@]}; i+=2)); do
    echo "  ${DOCKER_GROUP_ARGS[i]} ${DOCKER_GROUP_ARGS[i+1]} \\"
  done
  if [ "${USER_MODE}" = "host" ]; then
    echo "  --user ${UID_VALUE}:${GID_VALUE} \\"
  fi
  echo "  -w /workspace/unified-sdk \\"
  echo "  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\"
  echo "  ${IMAGE_NAME}:${TAG}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
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
    --optimum-rbln-version)
      [ -z "$2" ] && { echo "[ERROR] --optimum-rbln-version requires a value"; exit 1; }
      OPTIMUM_RBLN_VERSION="$2"; shift 2 ;;
    --vllm-rbln-version)
      [ -z "$2" ] && { echo "[ERROR] --vllm-rbln-version requires a value"; exit 1; }
      VLLM_RBLN_VERSION="$2"; shift 2 ;;
    --pytorch-index-url)
      [ -z "$2" ] && { echo "[ERROR] --pytorch-index-url requires a value"; exit 1; }
      PYTORCH_INDEX_URL="$2"; shift 2 ;;
    --cdi-device)
      [ -z "$2" ] && { echo "[ERROR] --cdi-device requires a value"; exit 1; }
      CDI_DEVICE="$2"; shift 2 ;;
    --user-mode)
      [ -z "$2" ] && { echo "[ERROR] --user-mode requires a value"; exit 1; }
      USER_MODE="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; print_usage; exit 1 ;;
  esac
done

case "${USER_MODE}" in
  root|host) ;;
  *)
    echo "[ERROR] --user-mode must be 'root' or 'host'"
    exit 1
    ;;
esac

[ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="rbln-only"
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
echo "  Dockerfile     : ${DOCKERFILE_PATH}"
echo "  Container name : ${CONTAINER_NAME}"
echo "  Workspace(repo): ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  Compiler ver.  : ${COMPILER_VERSION}"
echo "  Optimum ver.   : ${OPTIMUM_RBLN_VERSION}"
echo "  vLLM ver.      : ${VLLM_RBLN_VERSION}"
echo "  PyTorch index  : ${PYTORCH_INDEX_URL}"
echo "  CDI device     : ${CDI_DEVICE:-auto}"
echo "  User mode      : ${USER_MODE}"
echo "  Host UID:GID   : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  --secret "id=netrc,src=${NETRC_PATH}" \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg REBEL_COMPILER_VERSION="${COMPILER_VERSION}" \
  --build-arg OPTIMUM_RBLN_VERSION="${OPTIMUM_RBLN_VERSION}" \
  --build-arg VLLM_RBLN_VERSION="${VLLM_RBLN_VERSION}" \
  --build-arg PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL}" \
  .

detect_runtime_mounts
detect_keep_groups_support
detect_device_group

echo "Build complete!"
echo ""
if [ "${CDI_SPEC_DETECTED}" -eq 0 ] && [ -z "${RBLN_CDI_DEVICE:-}" ]; then
  echo "[WARN] CDI spec was not detected under /var/run/cdi or /etc/cdi."
  echo "       RBLN official user guide recommends Container Toolkit CDI:"
  echo "         sudo rbln-ctk cdi generate"
  echo "         sudo rbln-ctk runtime configure --runtime docker"
  echo "         sudo systemctl restart docker"
  echo "       This script will still print a CDI-based docker run example using rebellions.ai/npu=all,"
  echo "       but the container may not receive RBLN libraries/tools until CDI is configured correctly."
  echo ""
fi
echo "[INFO] Using RBLN CDI device handle: ${CDI_DEVICE}"
if [ "${CDI_SPEC_DETECTED}" -eq 1 ] && [ -n "${CDI_SPEC_HINT}" ]; then
  echo "[INFO] Detected CDI configuration via: ${CDI_SPEC_HINT}"
fi
if [ "${#DOCKER_GROUP_ARGS[@]}" -gt 0 ]; then
  echo "[INFO] Propagating container group options: ${DOCKER_GROUP_ARGS[*]}"
else
  echo "[WARN] No supplemental device group could be inferred from /dev/rbln*, /dev/rebellions*, or /dev/atom*."
  echo "       If rebel.npu_is_available() stays False, inspect host-side device ownership with:"
  echo "         ls -l /dev/rbln* /dev/rebellions* /dev/atom* 2>/dev/null || true"
fi
echo ""

echo "Run container with:"
print_run_hint

echo ""
echo "Sanity check inside container:"
echo "  command -v rbln-smi && rbln-smi || true"
echo "  python3 -c \"import unified_sdk, rebel; print('OK')\""
echo "  RBLN_DEVICES=0 python3 -c \"import rebel; print('npu_is_available=', rebel.npu_is_available())\""
echo "  python3 -c \"import torch, torchvision, rebel; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('rebel=', getattr(rebel, '__version__', 'unknown'))\""
echo "  python3 -c \"import optimum.rbln; import vllm; print('optimum-rbln/vllm-rbln OK')\""
echo "  RBLN_DEVICES=0 python3 examples/run_rbln_build.py"
