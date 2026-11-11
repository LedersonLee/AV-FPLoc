# Reproducibility and execution record

Study `reflection_timing_audit_20261111`, 2026-11-11. This is Experiment1B, distinct from preserved Experiment1. No trained model/checkpoint or HSSD evaluation dependency was introduced. Code/Docs repository is AV-FPLoc; work branch `refactor/hssd-two-branch-fusion`. Raw results remain outside Git at `/home/jeongeonlee/track1/experiments/reflection_timing_audit_20261111`. Original RIRs remain in sibling `reflection_observability_20261111`.

## Actual executed code

| Stage | Actual code Commit SHA | Notes |
|---|---|---|
| Documented C API main binary v3 | `4a9070a5faa60bb9f26c60f78f78edac862b3ffb` | Core C++ implementation; first invalid/corrected pilot preceded commit and was dirty. Pilot is cost/diagnostic only. |
| Corrected runner / backend_v2 | `f5d16dbbdd6b02826143c6b9136b06588fc7ecbb` | 48contexts /276mono outputs; decimal prefix fix |
| Authoritative old/new estimator analysis | `f86af5d81ef89ee46c56ae9e9ce6529d58a81a17` | 96old hashes;60peak replays;1,728 old channel×threshold rows;276backend rows |
| Stationary / phase audit | `2b0fe744a254055df32accf08fc0a1a3c960ab06` | No waveform time alignment |
| Material isolation v6 binary/runner | `46c1ddfdea29e30f6230126752b27a735f9b84d1` | specular_v3:24contexts /144mono outputs |
| Final specular analysis and verifier | `65e7eb1588fd7d99a84f6cb5da201dff02269ab1` | Fixed estimators, material-only differences, OutcomeC |
| Viewer | `0ae34bac9fc56ba721087a7d31c02868fc3cc890` | Builder first at e589294; moved-distance label correction at this revision |

The repository had two pre-existing unrelated dirty files, `scripts/gibson_runtime/global_acoustic_likelihood_gpu.py` and `scripts/gibson_runtime/render_f3loc_echoscan_rirs.py`; neither is used/committed by this audit. Stage code files were committed before the corresponding authoritative runs. Do not substitute a Docs commit for these execution SHAs. Full file, input, binary and audit-text hashes: [manifest](reproducibility_manifest.json). Pilot source was not a clean-clone run; later valid main runs provide committed implementation.

## Renderer and environment

- Docker tag `track1-soundspaces-echoloc:latest`; immutable image ID `sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1`.
- Habitat source revision `4f61e321477708fa606fbd8f42b4bef41d67c672`; RLR package revision `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`.
- Installed library `/opt/conda/envs/soundspaces/lib/python3.9/site-packages/habitat_sim/_ext/libRLRAudioPropagation.so`, SHA256 `31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4`.
- Header root `/opt/src/habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/headers`; bundled material config under adjacent `data/mp3d_material_config.json`.
- Host analysis Python3.10.12, NumPy1.21.5, SciPy1.8.0; Linux6.8.0-138 x86_64. Native renderer runner uses container Python3.9 and C++14 compiler. Host Chrome headless renders offline HTML.
- All rendering containers: network none, CPU quota4 of20host CPUs(20%, below80%), memory8GiB; RLR threadCount2. No GPU and no Node2 job. Concurrent unrelated workloads are not claimed as this audit's usage.
- Public RLR header does not expose a per-path acoustic GT/histogram accessor. Ambient c=343m/s is the analytic physical reference; this study does not independently recover an internal ambient-speed constant from the closed binary.

## RIR configuration / data provenance

48kHz pressure float32; unitScale1; indirect20,000 rays/source512 rays; indirect/source depth50; diffraction=true; transmission=false; direct/indirect=true except explicit suppression arms. Temporal coherence=false except named arms. Source radius default0; listener radius default0.1m or explicit0. Mesh: double-sided wall x±12,y−8..12,z0, moved−1m; distant floor y−20. Six mono listeners lie on0.05m ring at source height1.5m, source=(0.17,1.5,d). `simultaneous` uses six listener objects in one simulation; other ring arms move a single mono listener sequentially. Stationary arms fix mic0 for5calls.

Default material is absorption0.1/scattering0.5/transmission0. Explicit default and specular controls load the same bundled default material record and assign label `audit_wall`; specular changes only scattering coefficient values to0. The copied medium density/speed/damping are unchanged, and the material speed field is not the air sound speed. All24custom JSON pairs and actual OBJ coefficients were checked.

Each primary backend arm has3fresh contexts at2/6m. Context is not a controlled independent RNG seed: some first responses match almost exactly. No seed independence or scene-level confidence is claimed. Recorded variable IR length is preserved in each receipt; analysis uses fixed1–80ms, with original sample0 unchanged. No truncation/padding enters timing estimators; display pads only when needed.

Original96RIR inputs and their settings remain in original receipts. No original assets/checkpoints were overwritten. The analytic fractional-reference files are explicitly named as analytic and are not renderer recordings. This work does not require HSSD raw data for replay of the new controlled audit; the old HSSD result is only cited, not rerun.

## Cost and pilot before scaling

Corrected one-ring pilot:0.03146s measured renderer scope,440,736bytes; predicted main suite about20MB. This pilot preceded the bounded suite. Main valid backend_v2:276mono outputs,18,203,812bytes, summed C++ elapsed0.9823s and summed per-process wall1.1625s. Specular_v3:144mono outputs,10,042,836bytes, C++0.6598s and process wall0.7726s. Valid suites total**420mono /28,246,648bytes**, about28.25MB decimal. Durations are receipt sums for these small procedural scenes, excluding compiler/Docker startup, analysis, browsing and human review; they are not HSSD throughput forecasts or CPU-hours. Analytic references, logs, pilots, failed runs and screenshots are additional local storage (~38MB total at reporting checkpoint). There was no large-scale rendering.

## Replay existing results

From repository root, with the retained local data present:

```bash
ROOT=/home/jeongeonlee/track1/experiments/reflection_timing_audit_20261111
CODE=scripts/research/track_a/reflection_timing_audit_20261111
python3 "$CODE/analysis.py" --root "$ROOT"
python3 "$CODE/channel_audit.py" --root "$ROOT"
python3 "$CODE/specular_analysis.py" --root "$ROOT"
python3 "$CODE/verify.py" --root "$ROOT"
python3 "$CODE/build_viewer.py" --root "$ROOT" --out /tmp/reflection-timing-replay.html
```

These recompute derived JSON at the study root; raw RIR files are read-only inputs. To preserve all derived historical copies as well, copy the study directory before replay. The manifest includes expected output hashes from the completed run; `analysis.json` includes elapsed seconds so replay hash can differ even if all values agree. No stochastic re-render is needed for numerical replay.

## Fresh rendering

Choose a new sibling directory; runner refuses an existing output directory. Original image and prior raw RIR data are dependencies, so repository clone alone is **not** a self-contained reproduction. The binary/image are not uploaded to Git.

```bash
REPO=/home/jeongeonlee/track1/AV-FPLoc
FRESH=/home/jeongeonlee/track1/experiments/reflection_timing_audit_fresh
mkdir "$FRESH"
docker run --rm --network none --cpus 4 --memory 8g \
  --user "$(id -u):$(id -g)" --entrypoint bash \
  -v "$REPO:/code:ro" -v "$FRESH:/output" \
  sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1 -c '
  g++ -O2 -Wno-deprecated-declarations -std=c++14 \
    -I/opt/src/habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/headers \
    /code/scripts/research/track_a/reflection_timing_audit_20261111/rlr_audit.cpp \
    -L/opt/conda/envs/soundspaces/lib/python3.9/site-packages/habitat_sim/_ext \
    -Wl,-rpath,/opt/conda/envs/soundspaces/lib/python3.9/site-packages/habitat_sim/_ext \
    -lRLRAudioPropagation -o /output/rlr_audit &&
  /opt/conda/envs/soundspaces/bin/python \
    /code/scripts/research/track_a/reflection_timing_audit_20261111/run_backend.py \
    --out /output/backend_v2 --executable /output/rlr_audit &&
  /opt/conda/envs/soundspaces/bin/python \
    /code/scripts/research/track_a/reflection_timing_audit_20261111/run_backend.py \
    --out /output/specular_v3 --executable /output/rlr_audit \
    --modes explicit_default,specular,specular_radius0,specular_moved'
```

Run the analysis commands with ROOT=FRESH; the historical Experiment1 directory must remain its sibling. Latest C++ includes the new modes and produces the same base configuration as v3, but stochastic raw hashes need not match. To compile exact historical binaries use the stage SHA in the table. No effective-speed/delay tuning is part of reproduction.

## 실패 시도와 정정

1. First pilot default-configuration call occurred before `thisSize` initialization, yielding explicit API error. Preserved local `compile_pilot.log`, initial pilot outputs/binary are not authoritative. Corrected pilot initializes size and checks return codes; valid main suite used this correction.
2. `backend_v1` runner used `Path.with_suffix` on `d2.0...` prefix, looked for the wrong receipt filename and stopped after one ring. Raw partial results/log retained; `f5d16db` fixes concatenation, new output `backend_v2` preserves old directory.
3. Added paired-band analysis initially introduced tuple-valued dictionary keys; failed `analysis_v2.log` retained. `f86af5d` corrected it, successful `analysis_v3.log` and final `analysis.json` are authoritative. Earlier `analysis_v1.json` remains local.
4. Initial mesh verifier counted OBJ source/listener debug markers as geometry and asserted. Corrected verifier stops at `# Listeners`, tests8actual vertices; failure `verify.log` retained. No renderer output was rewritten.
5. `specular_v1` manual material JSON lacked damping; `specular_v2` then lacked density. Both returned API errors and were excluded. Final runner copies full bundled Default and changes only scattering; valid `specular_v3` manifest/export passes. Failed source commits `8bccf2e` and `bd32862` remain in history.
6. Full-RIR causal band onsets falsely detect direct-sound filter tail on removed walls. Paired waveform difference before band filter separates this artifact but is nondeployable and still does not establish physical alignment. Both methods and all thresholds remain in the report.
7. Initial default-only verifier concluded OutcomeB before scattering intervention. Final `verify_final.log`/Docs verification concludes OutcomeC; prior failed/intermediate records remain local. OutcomeC adds a measured setting dependency; it does not erase wall observability or prior limitations.

## Verification / archival limits

96prior RIR hashes;60exact old peak replays;36independent Fermat path checks;276main+144material mono hashes;72acoustic exports;6independent convolution-based onset replays;24custom material/config comparisons. Twelve default estimators fail the frozen physical gate; no HSSD re-execution was authorized by that gate. Viewer32recordings×6channels is browser-checked (192views) and visually reviewed. Reduced signed min/max/envelope display aggregates are embedded; raw waves, binary/library, source archives and logs remain local. These constraints prevent pretending a public clone alone includes the renderer/data.
