#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

BACKEND="trt"
FLAVOR="vision"
PASSTHROUGH=()

print_usage() {
  cat <<'EOF'
Usage: ./build.sh [--backend <qb|rbln|warboy|rngd|trt>] [--flavor <vision|llm>] [backend-script options...]

Top-level dispatcher:
  --backend qb       Mobilint QB vision/LLM environment
  --backend rbln     Rebellions RBLN vision/LLM environment
  --backend warboy   Furiosa Warboy vision environment
  --backend rngd     Furiosa RNGD LLM environment
  --backend trt      NVIDIA TensorRT environment

TensorRT flavor:
  --flavor vision    TensorRT vision image (default)
  --flavor llm       TensorRT-LLM image

Legacy aliases:
  --target tensorrt    -> --backend trt
  --target rebellions  -> --backend rbln
  --target furiosa     -> error (choose warboy or rngd explicitly)

Examples:
  ./build.sh --backend qb
  ./build.sh --backend rbln --workspace /path/to/repo
  ./build.sh --backend warboy
  ./build.sh --backend rngd
  ./build.sh --backend trt --flavor vision
  ./build.sh --backend trt --flavor llm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      [ -z "$2" ] && { echo "[ERROR] --backend requires a value"; exit 1; }
      BACKEND="$2"
      shift 2
      ;;
    --flavor)
      [ -z "$2" ] && { echo "[ERROR] --flavor requires a value"; exit 1; }
      FLAVOR="$2"
      shift 2
      ;;
    --target)
      [ -z "$2" ] && { echo "[ERROR] --target requires a value"; exit 1; }
      case "$2" in
        tensorrt) BACKEND="trt" ;;
        rebellions) BACKEND="rbln" ;;
        furiosa)
          echo "[ERROR] Legacy target 'furiosa' is ambiguous in main. Use --backend warboy or --backend rngd."
          exit 1
          ;;
        *)
          echo "[ERROR] Unsupported legacy target: $2"
          exit 1
          ;;
      esac
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

case "${BACKEND}" in
  qb)
    TARGET_SCRIPT="${PROJECT_ROOT}/scripts/build_qb.sh"
    ;;
  rbln)
    TARGET_SCRIPT="${PROJECT_ROOT}/scripts/build_rbln.sh"
    ;;
  warboy)
    TARGET_SCRIPT="${PROJECT_ROOT}/scripts/build_warboy.sh"
    ;;
  rngd)
    TARGET_SCRIPT="${PROJECT_ROOT}/scripts/build_rngd.sh"
    ;;
  trt)
    case "${FLAVOR}" in
      vision|llm) ;;
      *)
        echo "[ERROR] --flavor must be one of: vision, llm"
        exit 1
        ;;
    esac
    TARGET_SCRIPT="${PROJECT_ROOT}/scripts/build_trt.sh"
    PASSTHROUGH=(--flavor "${FLAVOR}" "${PASSTHROUGH[@]}")
    ;;
  *)
    echo "[ERROR] Unsupported backend: ${BACKEND}"
    echo ""
    print_usage
    exit 1
    ;;
esac

if [ ! -x "${TARGET_SCRIPT}" ]; then
  echo "[ERROR] Target build script is missing or not executable: ${TARGET_SCRIPT}"
  exit 1
fi

exec "${TARGET_SCRIPT}" "${PASSTHROUGH[@]}"
