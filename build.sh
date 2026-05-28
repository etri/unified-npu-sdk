#!/bin/bash
set -e

# =====================================
# unified-sdk (RBLN) build script
# =====================================

IMAGE_NAME="unified-sdk"
TAG="rbln"
CONTAINER_NAME=""
WORKSPACE_DIR=""
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

print_usage() {
  echo "Usage: $0 [--name <container_name>] [--workspace <repo_path>]"
  echo ""
  echo "Options:"
  echo "  --name        Container name (default: ${IMAGE_NAME}_${TAG}_dev)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  -h, --help    Show this help message"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [ -z "$2" ] && { echo "[ERROR] --name requires a value"; exit 1; }
      CONTAINER_NAME="$2"; shift 2 ;;
    --workspace)
      [ -z "$2" ] && { echo "[ERROR] --workspace requires a value"; exit 1; }
      WORKSPACE_DIR="$2"; shift 2 ;;
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
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  --secret "id=netrc,src=${NETRC_PATH}" \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  .

echo "Build complete!"
echo ""
echo "Run container with (RBLN device mount example):"
cat <<EOF
docker run -it --security-opt seccomp=unconfined \\
  --name ${CONTAINER_NAME} \\
  --device /dev/rsdo:/dev/rsdo \\
  --device /dev/rbln0:/dev/rbln0 \\
  --device /dev/rbln1:/dev/rbln1 \\
  -v ${WORKSPACE_DIR}:/workspace/unified-sdk \\
  -v /usr/local/bin/rbln-smi:/usr/local/bin/rbln-smi \\
  -v /usr/local/bin/rbln-stat:/usr/local/bin/rbln-stat \\
  ${IMAGE_NAME}:${TAG}
EOF

echo ""
echo "Sanity check inside container:"
echo "  rbln-smi"
echo "  python -c \"import unified_sdk, rebel; print('OK')\""
