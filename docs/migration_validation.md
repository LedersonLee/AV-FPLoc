# HSSD packaging validation — 2026-10-26 KST

Scope: two independent branches and fixed/adaptive scalar fusion. No new
scientific experiment or Test-set evaluation was performed.

## Preserved behavior

* Original `/home/jeongeonlee/track1` execution sources were left unchanged.
* The two pre-existing Gibson edits were excluded from all migration commits.
* Legacy source snapshot hashes are in `experiments/legacy/source_manifest.json`.
* E/C/N extraction on an actual Train RIR and the existing acoustic checkpoint
  output matched original code exactly on CPU.
* Core14/processed58 input ordering and processed58 numerical values matched
  the original code exactly on the deterministic diagnostic input.
* Existing 128→64 gate checkpoint loaded strictly and outputs matched exactly.
* A newly fetched pinned UnLoc/DepthAnything checkout plus the tracked patch
  ran actual Train RGB through the epoch20 checkpoint on GPU (40 depth/scale
  outputs). No access to the old source checkout was needed for that run.
* Portable dense score export and adaptive fusion ran on three Train queries.

## Tests and artifacts

The package test suite covers score normalization/MAP equivalence, invalid
weights, feature schemas, continuous-weight winner intervals, candidate/query
alignment, ray ordering, Repair/Harm, main/experiment dependency separation,
checkpoint roundtrip, split leakage rejection and one-epoch synthetic gate
training. All 15 tests passed in the existing Torch environment.

Five shared artifact files were checked, including checkpoint SHA256 values.
The epoch20 Vision checkpoint and pretrained DA weights were absent at the
documented File2 paths; they were copied without overwriting existing files,
and destination hashes matched the sources. See `configs/datasets/hssd_release.json`.
Small Train-only smoke banks are under `HSSD/releases/two_branch_fusion_v1/`.

## Limits

* This is not a new full Train/Val/Test performance evaluation.
* Old experiment scripts are a provenance snapshot, not all independently
  runnable on a clean machine; their old cache paths/imports are documented.
* The portable fusion trainer is explicitly a new reusable entry point; its
  per-query update schedule is not misrepresented as the old batch16 schedule.
* Full scene caches remain where originally generated; only essential weights
  and a bounded reproduction example were staged to File2.
* Source installation is separate from permission to use HSSD/upstream assets.
