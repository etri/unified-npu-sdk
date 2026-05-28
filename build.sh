#!/bin/bash
set -e

# =====================================
# unified-sdk (RBLN) build script
# =====================================

IMAGE_NAME="unified-sdk"
TAG="rbln"
CONTAINER_NAME=""
WORKSPACE_DIR=""
COMPILER_VERSION="${REBEL_COMPILER_VERSION:-0.9.4}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_DEVICE_ARGS=()
DOCKER_TOOL_MOUNTS=()

print_usage() {
  echo "Usage: $0 [--name <container_name>] [--workspace <repo_path>] [--compiler-version <version>]"
  echo ""
  echo "Options:"
  echo "  --name        Container name (default: ${IMAGE_NAME}_${TAG}_dev)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  --compiler-version  rebel-compiler version to install during docker build"
  echo "                (default: ${COMPILER_VERSION})"
  echo "  -h, --help    Show this help message"
}

detect_runtime_mounts() {
  if [ -e /dev/rsdo ]; then
    DOCKER_DEVICE_ARGS+=( "--device" "/dev/rsdo:/dev/rsdo" )
  fi

  for dev in /dev/rbln*; do
    if [ -c "${dev}" ]; then
      DOCKER_DEVICE_ARGS+=( "--device" "${dev}:${dev}" )
    fi
  done

  for tool in /usr/local/bin/rbln-smi /usr/local/bin/rbln-stat; do
    if [ -f "${tool}" ]; then
      DOCKER_TOOL_MOUNTS+=( "-v" "${tool}:${tool}" )
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
    --compiler-version)
      [ -z "$2" ] && { echo "[ERROR] --compiler-version requires a value"; exit 1; }
      COMPILER_VERSION="$2"; shift 2 ;;
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
echo "  Compiler ver.  : ${COMPILER_VERSION}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  --secret "id=netrc,src=${NETRC_PATH}" \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg REBEL_COMPILER_VERSION="${COMPILER_VERSION}" \
  .

detect_runtime_mounts

echo "Build complete!"
echo ""
if [ ${#DOCKER_DEVICE_ARGS[@]} -eq 0 ]; then
  echo "[WARN] No RBLN device nodes were detected on this host."
  echo "       Add the appropriate --device arguments manually before running the container."
  echo ""
fi

echo "Run container with:"
print_run_hint

echo ""
echo "Sanity check inside container:"
echo "  rbln-smi"
echo "  python3 -c \"import unified_sdk, rebel; print('OK')\""
echo "  RBLN_DEVICES=0 python3 examples/run_rbln_build.py"
