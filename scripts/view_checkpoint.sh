#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BAZEL="${BAZEL:-bazel}"
BUILD_CONFIG="${BUILD_CONFIG:---config=opt}"
ROM_PATH="${ROM_PATH:-roms/breakout.bin}"
CHECKPOINT="${CHECKPOINT:-checkpoints/train/latest.pt}"
VIDEO_DIR="${VIDEO_DIR:-videos/view}"
CONFIG="${CONFIG:-configs/v0.yaml}"
EPISODES="${EPISODES:-3}"

if [[ ! -f "$ROM_PATH" ]]; then
  echo "Missing ROM: $ROM_PATH" >&2
  echo "Put breakout.bin under roms/ or set ROM_PATH=/path/to/breakout.bin." >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  echo "Train with checkpointing enabled, or set CHECKPOINT=/path/to/latest.pt." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Missing ffmpeg on PATH. Install ffmpeg before recording videos." >&2
  exit 1
fi

"$BAZEL" run "$BUILD_CONFIG" //src/bin:train -- \
  view \
  "$ROM_PATH" \
  "$CHECKPOINT" \
  "$VIDEO_DIR" \
  "$CONFIG" \
  "$EPISODES"
