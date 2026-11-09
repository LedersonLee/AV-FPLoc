# Independent Track B: initial development implementation

Input: RGB, six-channel observed RIR, structural floorplan and calibration. No
reference acoustic bank or Track A model import. The scorer exhaustively evaluates
the supplied map-only bank, then extracts raw/diverse Top20. This is an initial
small-CNN feasibility implementation, not the completed PC-TriLoc protocol.

The native map plane is X/Z with yaw toward +Z. A forward-left-up world is
(X,-Z) with negated yaw. Calibration `origin`, `spacing`, `map_angle` maps the
raster to scoring coordinates. Black pixels are blocked/unknown, not trusted
physical walls. The initial FP representation is a rotated 3x3 feature stencil
at 1m offsets, not a physical echo model or long-range ray matcher.

`DenseScorer.forward(rgb, rir, floorplan, calibration, candidate_grid,
chunk_size=512)` has no labels, scene IDs, or reference RIR inputs. All branches
are trainable; the initial raw 6-channel waveform encoder preserves channels.
The small pilot uses the first 8192 samples and omits late reverberation. The
camera is fixed to the audited dataset rig; varying camera intrinsics are not
supported yet. Concat/noninteraction are architectural diagnostics, not certified
active-parameter matched baselines. Do not claim H-B1–H-B6 from smoke tests.

Run inside the existing `avfploc:cu124-clean` image with repository at `/work`:

```sh
python -m pytest -q track_b/tests -p no:cacheprovider
python -m track_b.scripts.audit_dataset --root /data --out /out/audit_v1
python -m track_b.scripts.smoke --out /out/smoke_v1.json
python -m track_b.scripts.pilot --root /data --audit /out/audit_v1 --out /out/pilot_v1
```

Use a fresh output directory for each audit/pilot. Mount dataset read-only;
keep manifests, waveforms, predictions and checkpoints outside Git. See the
canonical research index for evidence and scope limitations. Pending: verified
long-range geometry, variable camera calibration, full QC, strict asset duplicate
checks, matched capacity and compute, full 0.25m/36 training, frozen/partial
regimes, Support-NMS, OOF discriminator, robust shifts and deployment costs.
