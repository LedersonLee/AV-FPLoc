# Reproduction

Use a fresh output directory; native waveform runners refuse overwrite. On the same host:

```bash
CODE=/home/jeongeonlee/track1/AV-FPLoc/experiments/semantic_sound_poc
OUT=/home/jeongeonlee/track1/experiments/semantic_sound_poc_replay
python3 "$CODE/run_prepare.py" --out "$OUT"
bash "$CODE/run_native.sh" "$OUT" geometry
bash "$CODE/run_native.sh" "$OUT" render validate_depth_clearance.py
bash "$CODE/run_native.sh" "$OUT" render
bash "$CODE/run_native.sh" "$OUT" render run_direct_probe.py
python3 "$CODE/run_oracle.py" --out "$OUT"  # preserve provisional result
python3 "$CODE/apply_depth_correction.py" --out "$OUT"
OUT="$OUT/validated"
python3 "$CODE/run_oracle.py" --out "$OUT"
python3 "$CODE/analyze_audio.py" --out "$OUT"
python3 "$CODE/verify_results.py" --out "$OUT"
python3 -m unittest discover -s "$CODE" -p test_sanity.py
python3 "$CODE/make_figures.py" --out "$OUT"
```

Native runtime image ID `sha256:6abd00f2d4bbab222532b6f1c001577afe947d7f501e61936066dcd1dcae722b`; root dataset mounted read-only, code read-only, only OUT writable, network none, CPU4/memory8GB. Source/pose schedule seeds42/43/44 deterministic; native RLR RNG only partially controllable so compare preserved file hashes for exact replay. Dependencies [requirements.txt](/home/jeongeonlee/track1/AV-FPLoc/experiments/semantic_sound_poc/requirements.txt) and image versions recorded in manifest. Do not reconstruct full original cache provenance from clone alone; cached candidates and historical observation NPZ are external required inputs.

Actual primary oracle/native render executed from `ee9e56cd9027011a4c08132928b06d09e352ce4e`; initial preparation/geometry were dirty-source runs with matching file content later committed there. Diagnostic sources initially dirty on ee9e56c; final analysis source SHA `0950b2d8fa2b44dc7befc6ebbd18634da428cda4` with per-file hashes in manifest. Corrected Oracle execution SHA recorded in correction.json.
