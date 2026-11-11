# Reproducibility

Actual code, raw/input hashes and costs: [manifest](reproducibility_manifest.json). Raw root: `/home/jeongeonlee/track1/experiments/material_observability_20261111`. Code: `scripts/research/track_a/material_observability_20261111/`. No model/checkpoint loaded or trained. Code/Docs branch `refactor/hssd-two-branch-fusion`.

## Actual execution revisions

| Stage | Code SHA |
|---|---|
| C++ renderer / frozen design | `aff5991760d232bcd22cb16c161b0adb7e12f399` |
| Raw packing runner | `50b51b3ba4cc83e3467879b32654b3d0a434ce46` |
| Main coefficient profile corrected after pilot | `47d674082023bf1435eab89e4f81bc0afd9d6370` |
| Initial feature/reliability/noise/HSSD analysis | `97b5d71013f42f15e372d2f76ebdd51bffae2399` |
| Final representative exception correction | `e54e7069ff7fed66b62813486e3e7123a1da1e81` |
| Independent input/coefficient/scientific verification | `239de7d4e122dda1886930425cb5b38f9427bac9` |
| Energy/shape/distance diagnostics | `9fdbe0ed4441701037ff13a09d35f37e06aa069a` |
| Final explorer template geometry | `8587ca34f59750e7f1ceb35931bd2054e7028a98` |

The compiled C++ binary is unchanged after initialrender; exactSHA256isrecorded locally. Existing2dirtyGibsonscriptswere notused/committed. FinalDocsSHAisreportedaftercommit,notinsertedintoitsownindex. The final selection rerun includes moved walls, which are still physically present. It changes exception presentation only; rules and metrics are unchanged. The prior `results_selection_v1.json` is retained.

## Environment and input contract

Image ID: `sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1`.
Installed RLR library SHA256: `31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4`.
Habitat source revision: `4f61e321477708fa606fbd8f42b4bef41d67c672`.
RLR package revision: `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`.

Rendering uses container Python3.9 and C++14. Analysis uses host Python3.10.12, NumPy1.21.5 and SciPy1.8.0. HSSD preparation reuses study-local trimesh3.23.5. No global dependencies changed and no neural checkpoint was loaded.

Each condition retains its material JSON, input mesh, acoustic OBJ export, renderer receipt, full float32 `rir.npz`, original channel lengths, per-channel byte hashes and runtime. Compression is lossless. The runner deletes only its own temporary `.f32` files after exact array-equality verification. Original RIRs, checkpoints and assets were not overwritten.

Three fresh rendering contexts are recorded per condition. The API does not provide a verified independent-seed contract, so repeated contexts are not treated as independent scenes or guaranteed independent random samples.

## Cost and resource limits

The six-ring pilot took2.522s and3,445,204 compressed bytes. Its forecast was approximately1.14GB and832s for the main suite. It is excluded from the main analysis because provisional frequency anchors were corrected before the main run.

The main suite contains1,980 rings /11,880 mono channels,995,245,983 compressed bytes and634.994s summed per-ring wall time. Exact HSSD cost for72 additional rings /432 channels is recorded in [hssd_summary.json](hssd_summary.json) and the manifest. Time sums exclude compilation, container startup, manifest rewriting, browsing and review; they are not forecasts for furnished HSSD rendering.

Each rendering container was limited to4of20 host CPUs,8GiB memory and two renderer threads. Main and HSSD runs overlapped at most8of20 CPUs, below the80% limit. Analysis used two BLAS threads. No GPU or Node2 was used.

The local `HTML-reports-2` reference checkout at commit `b79e31c6bf1e09ad5d5785f76647a36ed83cae1f` is approximately1.5GB and is not experiment RIR storage. It is not committed here. Its semantic-compatibility viewer informed the light panels, paired views, scene controls and provenance presentation. Historical Test predictions were not used for fitting.

## Replay saved results

Run from the AV-FPLoc repository root:

```bash
ROOT=/home/jeongeonlee/track1/experiments/material_observability_20261111
CODE=scripts/research/track_a/material_observability_20261111
OPENBLAS_NUM_THREADS=2 python3 "$CODE/analyze.py" --root "$ROOT"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/results.py" --root "$ROOT"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/auxiliary.py" --root "$ROOT"
python3 "$CODE/diagnostics.py" --root "$ROOT"
python3 "$CODE/measurement_summary.py" --root "$ROOT"
python3 "$CODE/verify.py" --root "$ROOT"
python3 "$CODE/build_explorer.py" --root "$ROOT" --out /tmp/material-explorer.html
python3 "$CODE/verify_explorer.py" --root "$ROOT" --html /tmp/material-explorer.html
```

These commands rewrite derived JSON only. Copy the study directory first if all historical derived snapshots must also be preserved. Raw inputs remain read-only. Exact source and input hashes are in the manifest. Prose in the final reports was edited for readability after draft generation.

## Fresh rendering

Choose a new directory. The rendering runners refuse an existing destination. A Git clone alone is not self-contained: the immutable renderer image, local HSSD assets and prior visual-candidate inputs are separate dependencies.

```bash
REPO=/home/jeongeonlee/track1/AV-FPLoc
FRESH=/home/jeongeonlee/track1/experiments/material_observability_fresh
mkdir "$FRESH"
docker run --rm --network none --cpus 4 --memory 8g \
  --user "$(id -u):$(id -g)" --entrypoint bash \
  -v "$REPO:/code:ro" -v "$FRESH:/output" \
  sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1 -c '
  g++ -O2 -std=c++14 -Wno-deprecated-declarations \
    -I/opt/src/habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/headers \
    /code/scripts/research/track_a/material_observability_20261111/render.cpp \
    -L/opt/conda/envs/soundspaces/lib/python3.9/site-packages/habitat_sim/_ext \
    -Wl,-rpath,/opt/conda/envs/soundspaces/lib/python3.9/site-packages/habitat_sim/_ext \
    -lRLRAudioPropagation -o /output/render &&
  /opt/conda/envs/soundspaces/bin/python \
    /code/scripts/research/track_a/material_observability_20261111/run.py \
    --out /output/controlled --exe /output/render'
```

For a pilot, add `--pilot` and use a different output directory. The current pilot uses corrected anchors; the historical provisional anchors are preserved at commit50b51b3.

For HSSD, run `prepare_hssd.py --out "$FRESH/hssd_inputs"` on the host, then `run_hssd.py --root /output --exe /output/render` in the same bounded container. Preparation currently references the declared local HSSD and prior candidate paths; on another host, update paths and verify hashes. Query metadata supplies the physical source pose, while window selection uses cached pure-visual candidate geometry without GT filtering. The source GLB, RGB and FP files are never written.

## Failed interpretations, corrections and retained limits

- The pilot used provisional125/500/2000/8000Hz anchors. Actual metric export revealed83.91/369.25/1624.91/7150.52Hz. The main design was corrected before full rendering; the pilot is retained separately.
- Initial exception selection checked only the `present` state and missed moved walls that remain physically present. The corrected selection adds `single6`. The default medoid and every scientific rule/threshold remain unchanged. Both selection JSON versions are retained locally.
- The first HTML was about30MB. The final version uses480early/160full extrema bins,16-bit display quantization,160EDC samples at.01dB and128float32 spectral samples. Waveform display error is bounded by peak/65534. Metrics use original float32 data. This is display compression, not RIR interpolation or synthesis; no raw wave archive is uploaded.
- The mandatory clean absorption grid had no wall-present raw amplitude-gate miss. Two misses occurred in the prespecified .99 stress condition at the moved7m wall. This is not a general false-negative rate for real materials.
- Reliability and calibration failed to generalize. High removed-wall FPR and noise vulnerability were preserved instead of retuning after evaluation. Single2-only fitting limits the causal interpretation of this failure.
- Internal acoustic path IDs and the coefficient interpolation kernel remain unavailable. Native HSSD vertical-face assignment is a proxy, not validated furnished semantic mapping. Real material identification and localization remain unverified.
-662recordings end before400ms, so late energy is censored. Feature totals use the shortest common channel support within a ring; viewer EDCs use each channel's native support. RT60 is shown only as a qualified T20 extrapolation.

## Statistical unit and exposure

The experiment has six designed poses in three geometry families and four poses in two reused HSSD Train scenes. It does not have thousands of independent scenes. Pose-effect ranges and context SD are reported without treating repeated renders as independent-scene confidence intervals.

No Test payload was used for method selection or evaluation. Cached HSSD candidates inherit historical visual-teacher Train116/Val19 and acoustic Train12/Val3 exposure. Fresh RIR generation does not remove prior scene exposure. No new neural model, GEOQ/fusion training, Experiment2/3, full135-scene regeneration or dense FP-RIR bank was performed.

Final explorer code: `a1b0e83730747df2497653d43fa132ac21a90688`; measurement-summary code: `8322884cd8e23241fd4b41a29c5afe8a841dbe90`. The raw-to-display verifier checked all324condition packets /1,944channel displays, and Chrome passed349control checks plus the two counterexample views.
