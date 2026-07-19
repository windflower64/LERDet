#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "This script must run on the aarch64 RK3566 board." >&2
    exit 1
}
command -v g++ >/dev/null || { echo "Missing g++." >&2; exit 1; }
pkg-config --exists opencv4 || { echo "Missing OpenCV development files." >&2; exit 1; }
[[ -f runtime/include/rknn_api.h ]] || { echo "Missing runtime/include/rknn_api.h." >&2; exit 1; }
[[ -f /usr/lib/librknnrt.so ]] || { echo "Missing /usr/lib/librknnrt.so." >&2; exit 1; }

mkdir -p bin outputs
cflags=(
    -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic
    $(pkg-config --cflags opencv4)
    -I"${root}/include" -I"${root}/runtime/include"
)
libs=(
    /usr/lib/librknnrt.so
    $(pkg-config --libs opencv4)
)

g++ "${cflags[@]}" src/stagdet_detector.cpp src/image_main.cpp "${libs[@]}" -o bin/lerdet_image
g++ "${cflags[@]}" src/stagdet_detector.cpp src/benchmark_main.cpp "${libs[@]}" -o bin/lerdet_benchmark
g++ "${cflags[@]}" src/stagdet_detector.cpp src/perf_probe_main.cpp "${libs[@]}" -o bin/lerdet_perf_probe

echo "Built bin/lerdet_image"
echo "Built bin/lerdet_benchmark"
echo "Built bin/lerdet_perf_probe"
