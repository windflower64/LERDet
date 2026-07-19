#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${LERDET_VIDEO_DATA_ROOT:-/userdata/datasets/stagdet_raw_videos}"
sequence="${1:-$(awk -F, 'NR == 2 {gsub(/\r/, "", $1); print $1}' "${dataset_root}/video_pairs.csv")}"
confidence="${2:-0.25}"
max_frames="${3:-0}"
nms_iou="${NMS_IOU:-0.70}"
model="${LERDET_MODEL:-}"
[[ -s "${model}" ]] || { echo "Missing RKNN model. Set LERDET_MODEL to the final .rknn file." >&2; exit 1; }

row="$(awk -F, -v id="${sequence}" 'NR > 1 && $1 == id {gsub(/\r/, ""); print; exit}' "${dataset_root}/video_pairs.csv")"
[[ -n "${row}" ]] || { echo "Sequence not found: ${sequence}" >&2; exit 1; }
IFS=, read -r sequence_id visible_rel infrared_rel visible_json infrared_json expected_frames time_offset_ms visible_positive_frames <<<"${row}"
output_dir="${root}/outputs/${sequence_id}_conf${confidence}"

exec "${root}/bin/stagdet_video_opencv" \
    "${model}" \
    "${dataset_root}/${visible_rel}" \
    "${dataset_root}/${infrared_rel}" \
    "${output_dir}" \
    "${confidence}" \
    "${nms_iou}" \
    "${max_frames}"
