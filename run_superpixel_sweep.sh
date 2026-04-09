#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN_DEFAULT="$ROOT_DIR/superpixel_annotator/superpixel_annotator_venv/bin/python"

if [[ -x "$PYTHON_BIN_DEFAULT" ]]; then
  PYTHON_BIN="$PYTHON_BIN_DEFAULT"
else
  PYTHON_BIN="python3"
fi

IMAGE_PATH="${1:-$ROOT_DIR/_quarter_run/input/train_01_q1.jpg}"
MASK_PATH="${2:-$ROOT_DIR/_quarter_run/input/train_01_q1.png}"
OUTPUT_DIR="${3:-$ROOT_DIR/_quarter_run/sweeps/default_100}"
SSN_WEIGHTS="${4:-$ROOT_DIR/best_model.pth}"
EXTRA_ARGS=("${@:5}")

if command -v sysctl >/dev/null 2>&1; then
  CPU_COUNT="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
else
  CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

if [[ "$CPU_COUNT" -lt 2 ]]; then
  SIMPLE_WORKERS=1
elif [[ "$CPU_COUNT" -lt 8 ]]; then
  SIMPLE_WORKERS=2
else
  SIMPLE_WORKERS=4
fi

SSN_WORKERS="${SSN_WORKERS:-1}"
SIMPLE_WORKERS="${SIMPLE_WORKERS_OVERRIDE:-$SIMPLE_WORKERS}"
RESIZE_SCALE="${RESIZE_SCALE:-1.0}"

exec "$PYTHON_BIN" "$ROOT_DIR/sweep_interactive_superpixels.py" \
  --image "$IMAGE_PATH" \
  --mask "$MASK_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --python-bin "$PYTHON_BIN" \
  --methods felzenszwalb,slic,ssn \
  --scribbles 100 \
  --save_every 20 \
  --seed 0 \
  --resize-scale "$RESIZE_SCALE" \
  --margin 2 \
  --border_margin 3 \
  --no_overlap \
  --sensitivity 1.8 \
  --simple-workers "$SIMPLE_WORKERS" \
  --ssn-workers "$SSN_WORKERS" \
  --felz-scales 200,400,800 \
  --felz-sigmas 0.5,1.0 \
  --felz-min-sizes 20,50,100 \
  --slic-n-segments 500,1000,2000 \
  --slic-compactnesses 10,20,30 \
  --slic-sigmas 0.0,1.0 \
  --ssn-weights "$SSN_WEIGHTS" \
  --ssn-nspix-list 200,500,800 \
  --ssn-fdim 20 \
  --ssn-niter-list 5 \
  --ssn-color-scales 0.26 \
  --ssn-pos-scales 2.5 \
  "${EXTRA_ARGS[@]}"
