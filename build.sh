#!/bin/bash
set -e

# =====================================
# unified-sdk (QB / Mobilint ARISE) build script
# =====================================

IMAGE_NAME="unified-sdk"
TAG="qb"
CONTAINER_NAME=""
WORKSPACE_DIR=""
BASE_IMAGE="${QB_BASE_IMAGE:-ubuntu:22.04}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
QB_DEVICE="${QB_DEVICE:-}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_DEVICE_ARGS=()
DOCKER_TOOL_MOUNTS=()

print_usage() {
  echo "Usage: $0 [--name <container_name>] [--workspace <repo_path>] [--base-image <image>] [--pytorch-index-url <url>] [--device <node>]"
  echo ""
  echo "Options:"
  echo "  --name        Container name (default: ${IMAGE_NAME}_${TAG}_dev)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  --base-image  Docker base image used for build"
  echo "                (default: ${BASE_IMAGE})"
  echo "  --pytorch-index-url  PyTorch wheel index used for torch/torchvision"
  echo "                (default: ${PYTORCH_INDEX_URL})"
  echo "  --device      Mobilint device node, e.g. /dev/aries0"
  echo "                (default: auto-detect /dev/aries* and /dev/arise*)"
  echo "  -h, --help    Show this help message"
}

detect_runtime_mounts() {
  if [ -n "${QB_DEVICE}" ]; then
    DOCKER_DEVICE_ARGS+=( "--device" "${QB_DEVICE}:${QB_DEVICE}" )
    return
  fi

  for dev in /dev/aries* /dev/arise*; do
    if [ -c "${dev}" ]; then
      DOCKER_DEVICE_ARGS+=( "--device" "${dev}:${dev}" )
    fi
  done

  TOOL_CANDIDATES=(
    "$(command -v mobilint-cli 2>/dev/null || true)"
    /usr/local/bin/mobilint-cli
    /usr/bin/mobilint-cli
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
    --device)
      [ -z "$2" ] && { echo "[ERROR] --device requires a value"; exit 1; }
      QB_DEVICE="$2"; shift 2 ;;
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

# Mobilint SDK 패키지 (qubee + qbruntime) - 필수
VENDOR_DIR="${PROJECT_ROOT}/vendor"
if ! ls "${VENDOR_DIR}"/*.whl >/dev/null 2>&1; then
  echo "[ERROR] Mobilint SDK wheels not found under: ${VENDOR_DIR}"
  echo ""
  echo "Place vendor-provided qubee(compiler) + qbruntime(QB-RUNTIME) wheels first:"
  echo "  cp /path/to/qubee-*.whl     ${VENDOR_DIR}/"
  echo "  cp /path/to/qbruntime-*.whl ${VENDOR_DIR}/"
  echo ""
  echo "See https://docs.mobilint.com/v1.2/en/introduction.html"
  exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile     : ${PROJECT_ROOT}/Dockerfile"
echo "  Container name : ${CONTAINER_NAME}"
echo "  Workspace(repo): ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  PyTorch index  : ${PYTORCH_INDEX_URL}"
echo "  Vendor wheels  : $(ls "${VENDOR_DIR}"/*.whl | xargs -n1 basename | paste -sd, -)"
echo "  Device         : ${QB_DEVICE:-auto}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL}" \
  .

detect_runtime_mounts

echo "Build complete!"
echo ""
if [ ${#DOCKER_DEVICE_ARGS[@]} -eq 0 ]; then
  echo "[WARN] No Mobilint device nodes were detected on this host."
  echo "       Expected at least one /dev/aries* or /dev/arise* character device."
  echo "       Pass one explicitly with --device /dev/aries0 if needed."
  echo ""
fi

echo "Run container with:"
print_run_hint

echo ""
echo "Sanity check inside container:"
echo "  command -v mobilint-cli && mobilint-cli status || true"
echo "  python3 -c \"import unified_sdk, qbruntime; print('OK')\""
echo "  python3 -c \"import qbruntime; from qbruntime import type as t; print('devices=', t.get_available_device_numbers())\""
echo "  python3 -c \"import qubee; print('qubee=', getattr(qubee, '__version__', 'unknown'))\""
echo "  python3 examples/run_qb_build.py --help"
