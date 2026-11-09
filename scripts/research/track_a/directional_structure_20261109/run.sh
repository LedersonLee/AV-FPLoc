#!/usr/bin/env bash
set -euo pipefail
STUDY=$(cd "$(dirname "$0")" && pwd)
docker run --rm --gpus all --network none --cpus 4 --memory 16g --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONPATH=/workspace:/workspace/AV-FPLoc/src:/workspace/experiments/astra_fusion_v2_20261107_0225/deps -v /home/jeongeonlee/track1:/workspace:ro -v "$STUDY:/output:rw" -v /file2:/file2:ro -v /ssd1:/ssd1:ro avfploc:cu124-clean python "/output/$@"
