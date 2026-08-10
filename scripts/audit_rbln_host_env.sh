#!/usr/bin/env bash
set -euo pipefail

SINCE_DATE="${1:-2026-08-01}"
OUTPUT_PATH="${2:-/tmp/rbln_host_audit_$(hostname)_$(date +%Y%m%d_%H%M%S).log}"
USERS=(etri wormhole)

exec > >(tee "${OUTPUT_PATH}") 2>&1

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

section "RBLN Host Audit"
echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "hostname: $(hostname)"
echo "since_date: ${SINCE_DATE}"
echo "output_path: ${OUTPUT_PATH}"

section "User Permissions"
for user in "${USERS[@]}"; do
  echo
  echo "### user: ${user}"
  run_cmd "id ${user}" id "${user}"
  run_cmd "id -nG ${user}" id -nG "${user}"
  run_cmd "sudo -l -U ${user}" sudo -l -U "${user}"
done

run_cmd "getent group sudo" getent group sudo
run_cmd "getent group docker" getent group docker

section "Current Host Runtime Snapshot"
run_cmd "uname -a" uname -a
run_cmd "which rbln-smi" which rbln-smi
run_cmd "which rbln-ctk" which rbln-ctk
run_cmd "which docker" which docker
run_cmd "which containerd" which containerd

run_cmd "rbln-smi" rbln-smi
run_cmd "rbln-ctk info" rbln-ctk info
run_cmd "rbln-ctk cdi list" rbln-ctk cdi list
run_cmd "docker version" docker version
run_cmd "docker info" docker info

section "Installed Packages"
run_cmd "dpkg -l filtered" bash -lc "dpkg -l | egrep 'rbln|rebel|docker|containerd|linux-image|linux-modules' || true"

section "Device Nodes and CDI Specs"
run_cmd "ls -l devices/specs" bash -lc "ls -l /dev/rbln* /dev/atom* /var/run/cdi/rbln.yaml /etc/cdi/rbln.yaml 2>/dev/null || true"
run_cmd "stat devices/specs" bash -lc "stat /dev/rbln* /dev/atom* /var/run/cdi/rbln.yaml /etc/cdi/rbln.yaml 2>/dev/null || true"
run_cmd "sha256sum cdi specs" bash -lc "sha256sum /var/run/cdi/rbln.yaml /etc/cdi/rbln.yaml 2>/dev/null || true"

section "APT History"
run_cmd "apt history filtered from ${SINCE_DATE}" bash -lc "grep -hE 'Start-Date|Commandline|Install:|Upgrade:|Remove:' /var/log/apt/history.log* | sed -n '/${SINCE_DATE}/,\$p'"
run_cmd "dpkg log filtered" bash -lc "zgrep -hE 'install|upgrade|remove|configure' /var/log/dpkg.log* | egrep 'rbln|rebel|docker|containerd|linux-image|linux-modules|cdi' || true"

section "Journal Logs"
run_cmd "journalctl system filtered" bash -lc "journalctl --since '${SINCE_DATE}' | egrep 'rbln|rebel|docker|containerd|cdi' || true"
run_cmd "journalctl kernel filtered" bash -lc "journalctl -k --since '${SINCE_DATE}' | egrep -i 'rbln|rebel|firmware|pci|iommu|container|cdi' || true"

section "Auth / Sudo Activity"
run_cmd "auth.log filtered" bash -lc "zgrep -hE 'sudo|COMMAND=' /var/log/auth.log* | egrep 'etri|wormhole|rbln|docker|containerd|apt|dpkg' || true"

section "Summary Hints"
cat <<'EOF'
- Look for recent updates to: rbln-container-toolkit, rebel-*, docker*, containerd*, linux-image*, linux-modules*.
- Compare CDI spec timestamps and checksums with a previously working host if available.
- If rbln-smi works but rebel.npu_is_available() stays False, suspect host toolkit/CDI/userspace mismatch more than raw device permission issues.
- If etri or wormhole belongs to sudo/docker, either account could plausibly have changed the VM host environment.
EOF

echo
echo "Audit log saved to: ${OUTPUT_PATH}"
