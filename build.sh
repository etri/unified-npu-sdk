#!/bin/bash
set -e

# =====================================
# unified-sdk (FuriosaAI RNGD / furiosa-llm) 빌드 스크립트
# =====================================

IMAGE_NAME="unified-sdk"
TAG="rngd"
CONTAINER_NAME=""
WORKSPACE_DIR=""
BASE_IMAGE="${RNGD_BASE_IMAGE:-ubuntu:22.04}"
FURIOSA_PIP_INDEX="${FURIOSA_PIP_INDEX:-}"
RNGD_DEVICE="${RNGD_DEVICE:-}"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_DEVICE_ARGS=()
DOCKER_TOOL_MOUNTS=()

print_usage() {
  echo "사용법: $0 [--name <container_name>] [--workspace <repo_path>] [--base-image <image>] [--furiosa-pip-index <url>] [--device <node>]"
  echo ""
  echo "옵션:"
  echo "  --name        컨테이너 이름 (기본: ${IMAGE_NAME}_${TAG}_dev)"
  echo "  --workspace   /workspace/unified-sdk 로 마운트할 호스트 repo 경로 (기본: 현재 프로젝트 루트)"
  echo "  --base-image  빌드에 사용할 Docker base image (기본: ${BASE_IMAGE})"
  echo "  --furiosa-pip-index  furiosa-llm 설치용 추가 pip 인덱스 (선택)"
  echo "  --device      RNGD 장치 노드, 예: /dev/rngd0 (기본: /dev/rngd*, /dev/npu* 자동 감지)"
  echo "  -h, --help    도움말 출력"
}

detect_runtime_mounts() {
  if [ -n "${RNGD_DEVICE}" ]; then
    DOCKER_DEVICE_ARGS+=( "--device" "${RNGD_DEVICE}:${RNGD_DEVICE}" )
    return
  fi

  for dev in /dev/rngd* /dev/npu*; do
    if [ -c "${dev}" ]; then
      DOCKER_DEVICE_ARGS+=( "--device" "${dev}:${dev}" )
    fi
  done

  TOOL_CANDIDATES=(
    "$(command -v furiosa-smi 2>/dev/null || true)"
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
      [ -z "$2" ] && { echo "[ERROR] --name 값이 필요합니다"; exit 1; }
      CONTAINER_NAME="$2"; shift 2 ;;
    --workspace)
      [ -z "$2" ] && { echo "[ERROR] --workspace 값이 필요합니다"; exit 1; }
      WORKSPACE_DIR="$2"; shift 2 ;;
    --base-image)
      [ -z "$2" ] && { echo "[ERROR] --base-image 값이 필요합니다"; exit 1; }
      BASE_IMAGE="$2"; shift 2 ;;
    --furiosa-pip-index)
      [ -z "$2" ] && { echo "[ERROR] --furiosa-pip-index 값이 필요합니다"; exit 1; }
      FURIOSA_PIP_INDEX="$2"; shift 2 ;;
    --device)
      [ -z "$2" ] && { echo "[ERROR] --device 값이 필요합니다"; exit 1; }
      RNGD_DEVICE="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "[ERROR] 알 수 없는 옵션: $1"; print_usage; exit 1 ;;
  esac
done

[ -z "${CONTAINER_NAME}" ] && CONTAINER_NAME="${IMAGE_NAME}_${TAG}_dev"
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
echo "  Furiosa index  : ${FURIOSA_PIP_INDEX:-공개 PyPI}"
echo "  장치           : ${RNGD_DEVICE:-auto}"
echo "  UID:GID        : ${UID_VALUE}:${GID_VALUE}"

cd "${PROJECT_ROOT}"

DOCKER_BUILDKIT=1 docker build \
  -f "${PROJECT_ROOT}/Dockerfile" \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UID="${UID_VALUE}" \
  --build-arg GID="${GID_VALUE}" \
  --build-arg FURIOSA_PIP_INDEX="${FURIOSA_PIP_INDEX}" \
  .

detect_runtime_mounts

echo "빌드 완료!"
echo ""
if [ ${#DOCKER_DEVICE_ARGS[@]} -eq 0 ]; then
  echo "[WARN] 이 호스트에서 RNGD 장치 노드를 찾지 못했습니다."
  echo "       /dev/rngd* 또는 /dev/npu* 문자 장치가 최소 1개 필요합니다(호스트 드라이버 전제)."
  echo "       필요하면 --device /dev/rngd0 으로 직접 지정하세요."
  echo ""
fi

echo "컨테이너 실행:"
print_run_hint

echo ""
echo "컨테이너 내부 점검:"
echo "  furiosa-smi info || true"
echo "  python3 -c \"import unified_sdk; from furiosa_llm import LLM, SamplingParams; print('OK')\""
echo "  python3 examples/run_rngd_infer.py --help"
