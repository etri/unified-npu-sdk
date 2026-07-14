#!/bin/bash
set -e

# =====================================
# unified-sdk (QB / Mobilint ARISE) build script
# =====================================

IMAGE_NAME="unified-sdk"
TAG="qb"
CONTAINER_NAME=""
WORKSPACE_DIR=""
BASE_IMAGE="${QB_BASE_IMAGE:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
QB_RUNTIME_PIP_SPEC="${QB_RUNTIME_PIP_SPEC:-mobilint-qb-runtime}"
QB_DEVICE="${QB_DEVICE:-}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_DEVICE_ARGS=()
COMPILER_WHEELS=()

infer_base_image_from_wheel() {
  local wheel version tag_version
  for wheel in "${COMPILER_WHEELS[@]}"; do
    wheel="$(basename "${wheel}")"
    if [[ "${wheel}" =~ ^qbcompiler-([0-9]+(\.[0-9]+)*)(\+[A-Za-z0-9._-]+)?- ]]; then
      version="${BASH_REMATCH[1]}"
      IFS='.' read -r -a _parts <<< "${version}"
      if [ ${#_parts[@]} -ge 2 ]; then
        tag_version="${_parts[0]}.${_parts[1]}"
      else
        tag_version="${_parts[0]}"
      fi
      echo "mobilint/qbcompiler:${tag_version}-cpu-ubuntu22.04"
      return 0
    fi
  done
  return 1
}

print_usage() {
  echo "Usage: $0 [-n <container_name>] [--workspace <repo_path>] [--base-image <image>] [--pytorch-index-url <url>] [--device <node>]"
  echo ""
  echo "Options:"
  echo "  -n, --name    Container name (default: qb-only)"
  echo "  --workspace   Host repo path to mount into /workspace/unified-sdk"
  echo "                (default: current project root)"
  echo "  --base-image  Docker base image used for build"
  echo "                (default: infer from qbcompiler wheel, e.g. mobilint/qbcompiler:1.2-cpu-ubuntu22.04)"
  echo "                Use mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04 for GPU-accelerated compile."
  echo "  --pytorch-index-url  PyTorch wheel index used for torch/torchvision"
  echo "                (default: ${PYTORCH_INDEX_URL})"
  echo "  --runtime-pip-spec  pip spec used for Mobilint QB runtime"
  echo "                (default: ${QB_RUNTIME_PIP_SPEC})"
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

}

print_run_hint() {
  echo "docker run -it --security-opt seccomp=unconfined \\"
  echo "  --name ${CONTAINER_NAME} \\"
  for ((i=0; i<${#DOCKER_DEVICE_ARGS[@]}; i+=2)); do
    echo "  ${DOCKER_DEVICE_ARGS[i]} ${DOCKER_DEVICE_ARGS[i+1]} \\"
  done
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
    --pytorch-index-url)
      [ -z "$2" ] && { echo "[ERROR] --pytorch-index-url requires a value"; exit 1; }
      PYTORCH_INDEX_URL="$2"; shift 2 ;;
    --runtime-pip-spec)
      [ -z "$2" ] && { echo "[ERROR] --runtime-pip-spec requires a value"; exit 1; }
      QB_RUNTIME_PIP_SPEC="$2"; shift 2 ;;
    --device)
      [ -z "$2" ] && { echo "[ERROR] --device requires a value"; exit 1; }
      QB_DEVICE="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; print_usage; exit 1 ;;
  esac
done

[ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="qb-only"
[ -z "${WORKSPACE_DIR}" ] && WORKSPACE_DIR="${PROJECT_ROOT}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
  echo "[ERROR] Workspace directory not found: ${WORKSPACE_DIR}"
  exit 1
fi
WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd)"

# Mobilint SDK 패키지 - compiler wheel만 vendor 제공
VENDOR_DIR="${PROJECT_ROOT}/vendor"
shopt -s nullglob
COMPILER_WHEELS=("${VENDOR_DIR}"/qbcompiler-*.whl "${VENDOR_DIR}"/qubee-*.whl)
shopt -u nullglob
if [ ${#COMPILER_WHEELS[@]} -eq 0 ]; then
  echo "[ERROR] Mobilint qb compiler wheel not found under: ${VENDOR_DIR}"
  echo ""
  echo "Place the vendor-provided qb compiler wheel first:"
  echo "  cp /path/to/qbcompiler-*.whl ${VENDOR_DIR}/"
  echo ""
  echo "QB runtime is installed inside the image via pip:"
  echo "  ${QB_RUNTIME_PIP_SPEC}"
  echo ""
  echo "See https://docs.mobilint.com/v1.3/en/introduction.html"
  exit 1
fi

if [ -z "${BASE_IMAGE}" ]; then
  if ! BASE_IMAGE="$(infer_base_image_from_wheel)"; then
    echo "[ERROR] Could not infer Mobilint qbcompiler base image from vendor wheel name."
    echo ""
    echo "Expected a filename like one of:"
    echo "  qbcompiler-1.1.2+aries2-py3-none-any.whl"
    echo "  qbcompiler-1.2.0-py3-none-any.whl"
    echo ""
    echo "Please pass the compiler image explicitly, for example:"
    echo "  ./build.sh --base-image mobilint/qbcompiler:1.1-cpu-ubuntu22.04"
    echo ""
    echo "See https://docs.mobilint.com/v1.3/en/installing_compiler.html"
    exit 1
  fi
fi

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile     : ${PROJECT_ROOT}/Dockerfile"
echo "  Container name : ${CONTAINER_NAME}"
echo "  Workspace(repo): ${WORKSPACE_DIR}"
echo "  Base image     : ${BASE_IMAGE}"
echo "  PyTorch index  : ${PYTORCH_INDEX_URL}"
echo "  Runtime (pip)  : ${QB_RUNTIME_PIP_SPEC}"
echo "  Compiler wheel : $(printf '%s\n' "${COMPILER_WHEELS[@]}" | xargs -n1 basename | paste -sd, -)"
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
  --build-arg QB_RUNTIME_PIP_SPEC="${QB_RUNTIME_PIP_SPEC}" \
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
echo "  python3 -c \"import importlib, pkgutil; m = next((importlib.import_module(n) for n in ('qubee', 'qbcompiler') if pkgutil.find_loader(n)), None); print('compiler_pkg=', getattr(m, '__name__', 'missing'), 'version=', getattr(m, '__version__', 'unknown') if m else 'n/a')\""
echo "  python3 examples/run_qb_build.py --help"
