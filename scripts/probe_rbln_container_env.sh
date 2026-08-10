#!/usr/bin/env bash
set -uo pipefail

LABEL=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      echo "Usage: $0 [--label <name>] [--output <path>]" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${OUTPUT_PATH}" ]]; then
  exec > >(tee "${OUTPUT_PATH}") 2>&1
fi

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_cmd() {
  local label="$1"
  shift
  echo
  echo "--- ${label}"
  echo "\$ $*"
  "$@" || true
}

section "RBLN Container Probe"
echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "label: ${LABEL:-unset}"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "RBLN_DEVICES: ${RBLN_DEVICES:-unset}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
echo "PYTHONPATH: ${PYTHONPATH:-unset}"

section "Basic Runtime"
run_cmd "id" id
run_cmd "python3 --version" python3 --version
run_cmd "env | grep -E 'RBLN|CUDA|LD_|PYTHON'" bash -lc "env | egrep 'RBLN|CUDA|LD_|PYTHON' || true"

section "Device / Tool Visibility"
run_cmd "which rbln-smi" which rbln-smi
run_cmd "which rbln-stat" which rbln-stat
run_cmd "ls -l /dev/rbln* /dev/rsd*" bash -lc "ls -l /dev/rbln* /dev/rsd* 2>/dev/null || true"
run_cmd "rbln-smi" rbln-smi

section "Python Probe"
run_cmd "rebel import/probe" python3 - <<'PY'
import importlib.util
import json
import os
import sys

result = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "rebel_spec": None,
    "rebel_import_ok": False,
    "rebel_file": None,
    "rebel_version": None,
    "npu_is_available": None,
    "exception": None,
}

spec = importlib.util.find_spec("rebel")
result["rebel_spec"] = None if spec is None else spec.origin

try:
    import rebel
    result["rebel_import_ok"] = True
    result["rebel_file"] = getattr(rebel, "__file__", None)
    result["rebel_version"] = getattr(rebel, "__version__", "unknown")
    try:
        result["npu_is_available"] = rebel.npu_is_available()
    except Exception as exc:
        result["exception"] = f"probe_error: {type(exc).__name__}: {exc}"
except Exception as exc:
    result["exception"] = f"import_error: {type(exc).__name__}: {exc}"

print(json.dumps(result, indent=2, sort_keys=True))
PY

section "Linked Libraries"
run_cmd "ldconfig -p | grep rbln" bash -lc "ldconfig -p 2>/dev/null | grep -i rbln || true"
run_cmd "ls -l /usr/local/lib/rbln" bash -lc "ls -l /usr/local/lib/rbln 2>/dev/null || true"
run_cmd "find /usr/local/lib/rbln -maxdepth 2 -type f | sort" bash -lc "find /usr/local/lib/rbln -maxdepth 2 -type f 2>/dev/null | sort || true"

section "Vendor Package Snapshot"
run_cmd "pip show rebel-compiler" python3 -m pip show rebel-compiler
run_cmd "pip freeze filtered" bash -lc "python3 -m pip freeze | egrep 'rebel|rbln|optimum|vllm|torch' || true"

echo
echo "Probe complete."
