#!/usr/bin/env bash
set -euo pipefail

echo "OpenCV version: $(pkg-config --modversion opencv4 2>/dev/null || echo unavailable)"
if command -v opencv_version >/dev/null; then
    opencv_version --verbose | grep -A8 -E 'Video I/O|FFMPEG|GStreamer' || true
fi
for tool in gst-launch-1.0 gst-inspect-1.0; do
    if command -v "${tool}" >/dev/null; then
        echo "${tool}: available"
    else
        echo "${tool}: unavailable"
    fi
done
if command -v gst-inspect-1.0 >/dev/null; then
    for plugin in qtdemux mpeg4videoparse mppvideodec avdec_mpeg4; do
        if gst-inspect-1.0 "${plugin}" >/dev/null 2>&1; then
            echo "GStreamer plugin ${plugin}: available"
        else
            echo "GStreamer plugin ${plugin}: unavailable"
        fi
    done
fi
