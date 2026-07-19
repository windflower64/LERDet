#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${LERDET_DATA_ROOT:-${root}/test_data}"
manifest="${data_root}/manifest.csv"
model="${LERDET_MODEL:-}"
[[ -s "${manifest}" ]] || { echo "Missing dataset manifest: ${manifest}" >&2; exit 1; }
[[ -s "${model}" ]] || { echo "Missing RKNN model. Set LERDET_MODEL to the final .rknn file." >&2; exit 1; }
name="${1:-$(awk -F, 'NR == 2 {print $1}' "${manifest}")}"

exec "${root}/bin/lerdet_image" \
    "${model}" \
    "${data_root}/visible/${name}.jpg" \
    "${data_root}/infrared/${name}.jpg" \
    "${manifest}"
