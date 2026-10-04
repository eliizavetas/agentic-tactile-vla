#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${BRAINCO_GRASP_BINARY:-${script_dir}/build/brainco_grasp_service}"
port="${BRAINCO_GRASP_PORT:?set BRAINCO_GRASP_PORT to the serial device of the hand}"
serial="${BRAINCO_GRASP_SERIAL:?set BRAINCO_GRASP_SERIAL to the serial number of the hand}"
socket_path="${BRAINCO_GRASP_SOCKET:-/tmp/brainco_grasp.sock}"
socket_group="${BRAINCO_GRASP_SOCKET_GROUP:-dialout}"
nominal_gate="${BRAINCO_GRASP_NOMINAL_GATE:?set BRAINCO_GRASP_NOMINAL_GATE to the H3 meta file of a passed hard run}"
gentle_gate="${BRAINCO_GRASP_GENTLE_GATE:?set BRAINCO_GRASP_GENTLE_GATE to the H3 meta file of a passed soft_low run}"
log_dir="${BRAINCO_GRASP_LOG_DIR:-${script_dir}/data_out/grasp_service}"

if [[ ${EUID} -ne 0 ]]; then
    echo "Run this launcher with sudo." >&2
    exit 2
fi
if [[ ! -x "${binary}" ]]; then
    echo "Service binary not found: ${binary}" >&2
    echo "Build it first: bash ${script_dir}/build_brainco_grasp_service.sh" >&2
    exit 2
fi
if [[ ! -e "${port}" ]]; then
    echo "Serial device not found: ${port}" >&2
    exit 2
fi
if [[ ! -f "${nominal_gate}" || ! -f "${gentle_gate}" ]]; then
    echo "Validated H3 gate files were not found." >&2
    echo "nominal: ${nominal_gate}" >&2
    echo "gentle:  ${gentle_gate}" >&2
    exit 2
fi
if systemctl is-active --quiet brainco_hand.service; then
    echo "brainco_hand.service is active. Stop it before starting the grasp service." >&2
    exit 3
fi
if pgrep -af brainco_hand_server >/dev/null; then
    echo "brainco_hand_server is still running. The serial path must have one owner." >&2
    exit 3
fi
if fuser "${port}" >/dev/null 2>&1; then
    echo "Serial device is already open: ${port}" >&2
    fuser -v "${port}" >&2 || true
    exit 3
fi

mkdir -p "${log_dir}"
chown root:"${socket_group}" "${log_dir}"
chmod 2775 "${log_dir}"
umask 0002

exec "${binary}" \
    --confirm-hardware \
    --port "${port}" \
    --expected-serial "${serial}" \
    --nominal-gate-meta "${nominal_gate}" \
    --gentle-gate-meta "${gentle_gate}" \
    --socket "${socket_path}" \
    --socket-group "${socket_group}" \
    --log-dir "${log_dir}"
