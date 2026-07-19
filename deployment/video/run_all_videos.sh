#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${LERDET_VIDEO_DATA_ROOT:-/userdata/datasets/stagdet_raw_videos}"
confidence="${1:-0.25}"
max_frames="${2:-0}"
max_sequences="${3:-0}"
count=0

while IFS=, read -r sequence_id visible_rel infrared_rel visible_json infrared_json expected_frames time_offset_ms visible_positive_frames; do
    [[ "${sequence_id}" == "sequence_id" || -z "${sequence_id}" ]] && continue
    sequence_id="${sequence_id//$'\r'/}"
    echo "Running ${sequence_id}"
    LERDET_VIDEO_DATA_ROOT="${dataset_root}" "${root}/run_video_pair.sh" "${sequence_id}" "${confidence}" "${max_frames}"
    count=$((count + 1))
    if [[ "${max_sequences}" -gt 0 && "${count}" -ge "${max_sequences}" ]]; then
        break
    fi
done < "${dataset_root}/video_pairs.csv"

echo "Completed ${count} video pairs"
