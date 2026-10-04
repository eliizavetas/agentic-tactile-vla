#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${script_dir}/build"

mkdir -p "${output_dir}"

cmake -S "${script_dir}" -B "${output_dir}" -DSTARK_SDK_DIR="${STARK_SDK_DIR:?set STARK_SDK_DIR to the BrainCo Stark SDK}"
cmake --build "${output_dir}" -j4

echo "Built ${output_dir}/brainco_grasp_service and the gate diagnostics"
