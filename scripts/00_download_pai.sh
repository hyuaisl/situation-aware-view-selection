#!/bin/bash
# Download PhysicalAI-AV validation chunks (cameras + ego motion).
# Paper: chunks 186-188 of the validation split (300 clips).
#
#   SVS_RECIPES=/path/to/alpamayo-recipes bash scripts/00_download_pai.sh 186-188
#   SVS_RECIPES=... EXTRA_CAMS=1 bash scripts/00_download_pai.sh 186-188   # + 3 cameras for Fig. 1 only
#
# SVS_RECIPES must point to a checkout that provides scripts/download_pai.py.
# The Xet transfer backend stalled for us, so it is disabled; failed runs are retried.
set -u
CHUNKS="${1:?usage: 00_download_pai.sh <chunk range, e.g. 186-188>}"
OUT="${SVS_WORK:-.}/data/pai"
CAMS="camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov"
if [ "${EXTRA_CAMS:-0}" = "1" ]; then
  CAMS="$CAMS camera_rear_left_70fov camera_rear_right_70fov camera_front_tele_30fov"
fi
for try in $(seq 1 20); do
  HF_HUB_DISABLE_XET=1 python "${SVS_RECIPES:?set SVS_RECIPES}/scripts/download_pai.py" --chunk-ids "$CHUNKS" \
    --camera $CAMS \
    --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions --labels egomotion \
    --output-dir "$OUT" && { echo "OK chunks $CHUNKS after try $try"; exit 0; }
  echo "retry $try"; sleep 10
done
echo "FAILED chunks $CHUNKS"; exit 1
