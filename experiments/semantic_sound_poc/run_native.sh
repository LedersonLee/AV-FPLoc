#!/usr/bin/env bash
set -euo pipefail
out="$1"
mode="$2"
script="${3:-run_acoustic.py}"
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm --network none --cpus 4 --memory 8g --user "$(id -u):$(id -g)" --entrypoint /opt/conda/envs/soundspaces/bin/python -e CUDA_VISIBLE_DEVICES=-1 -e NVIDIA_VISIBLE_DEVICES=void -e EGL_PLATFORM=surfaceless -e LIBGL_ALWAYS_SOFTWARE=1 -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 -e LP_NUM_THREADS=1 -e MESA_SHADER_CACHE_DISABLE=true -e MAGNUM_LOG=quiet -e HABITAT_SIM_LOG=quiet -e PYTHONDONTWRITEBYTECODE=1 -v /file2:/file2:ro -v "$code_dir:/code:ro" -v "$out:/output:rw" -w /output track1-soundspaces-echoloc:ktx-hssd-20261015 "/code/$script" --out /output --mode "$mode"
