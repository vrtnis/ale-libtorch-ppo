#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BAZEL="${BAZEL:-bazel}"
BUILD_CONFIG="${BUILD_CONFIG:---config=opt}"
ROM_PATH="${ROM_PATH:-roms/breakout.bin}"
LOG_DIR="${LOG_DIR:-logs/debug}"
VIDEO_DIR="${VIDEO_DIR:-videos/debug}"
RUN_NAME="${RUN_NAME:-debug}"
CONFIG="${CONFIG:-configs/debug.yaml}"

if [[ ! -f "$ROM_PATH" ]]; then
  echo "Missing ROM: $ROM_PATH" >&2
  echo "Put breakout.bin under roms/ or set ROM_PATH=/path/to/breakout.bin." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Missing ffmpeg on PATH. Install ffmpeg before recording videos." >&2
  exit 1
fi

"$BAZEL" run "$BUILD_CONFIG" //src/bin:train -- \
  "$ROM_PATH" \
  "$LOG_DIR" \
  "$VIDEO_DIR" \
  "$RUN_NAME" \
  "$CONFIG"
