# Track A pipeline diagnostic exporter

This is a frozen inference and display workflow, not a training experiment.
`export.py` imports the historical B2, Support-NMS and GEOQ inference functions,
checks cached input hashes and candidate replay, and exports only derived display
assets. The two original unrelated dirty runtime scripts are not used or changed.

The runtime requires the existing private local artifacts under `experiments/`,
the HSSD asset mount, and Docker image `avfploc:cu124-clean`. This source snapshot
alone does **not** make the historical models/data reproducible from a clean clone.
Source hashes, input receipts and checkpoint hashes are embedded in the manifest.
Historical inference source Git SHAs were not recorded; hashes identify exact files.

Create a fresh output directory, copy `export.py` there, and run the command below.
Use `--pilot` first (three scenes), validate it, then run without `--pilot` to expand
the same viewer output to the approved New12 reused development scope. No Test17
observations, raw RIR rendering, gradient computation or optimizer updates occur.

```bash
# Set these to the existing workspace and a fresh output folder.
WORKSPACE_ROOT=/home/jeongeonlee/track1
VIEWER_OUTPUT="$WORKSPACE_ROOT/experiments/pipeline_viewer_20261110"
docker run --rm --gpus all --network none --cpus 4 --memory 16g \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace:/workspace/AV-FPLoc/src:/workspace/experiments/astra_fusion_v2_20261107_0225/deps \
  -v "$WORKSPACE_ROOT:/workspace:ro" -v "$VIEWER_OUTPUT:/output:rw" \
  -v /file2:/file2:ro -v /ssd1:/ssd1:ro \
  avfploc:cu124-clean python /output/export.py --pilot
```

Copy `index.html`, `viewer.js`, and `NOTICE.md` into `<output>/public/` and copy
`replay_validation.json` there after each export. Serve this folder with an HTTP
server; `file://` cannot fetch gzip assets. Run:

```bash
python3 validate.py <output>/public
python3 browser_qa.py http://localhost:8769 <output>/browser-qa
```

The browser integration test requires Playwright and local Chromium; its executable
path can be adapted to the host. The exporter uses NumPy, PyTorch, SciPy, joblib and
Pillow from the historical image. QA dependencies are separate and never published.

After verification, add the source snapshot commit SHA plus execution base SHA and
dirty source hash to manifest provenance. Publish **only** the public display folder
under the authorized HTML repository's non-main branch. Never publish the output
root, input data, QA dependencies, geometry cache or raw logs.
