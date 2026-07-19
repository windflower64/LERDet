#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

required=(
    runtime/include/rknn_api.h
    include/stagdet_detector.hpp
    src/stagdet_detector.cpp
    src/image_main.cpp
    src/benchmark_main.cpp
    src/perf_probe_main.cpp
    labels.txt
    build_on_board.sh
    run_image.sh
    run_benchmark.sh
    run_perf_probe.sh
)
for path in "${required[@]}"; do
    [[ -s "${path}" ]] || { echo "Missing or empty: ${path}" >&2; exit 1; }
done

for script in build_on_board.sh run_image.sh run_benchmark.sh run_perf_probe.sh verify_package.sh; do
    bash -n "${script}"
done
echo "Package verification passed. Dataset and model are external; set LERDET_DATA_ROOT and LERDET_MODEL before running."
