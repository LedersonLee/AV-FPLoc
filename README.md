# AV-FPLoc

Research scaffold for active acoustic-visual floorplan localization under visual degradation.

## Status

The repository currently defines the public method boundary and design contracts. The acoustic observation model, likelihood fusion, and evaluation pipeline are not implemented yet.

## Research goal

Estimate a single-observation 2D floorplan pose posterior by combining visual structural cues with active acoustic geometry cues. The primary question is whether acoustic geometry reduces the structural ambiguity that remains when visual observations are low-information or visually degraded.

## Repository layout

```text
track1_core/  Future implementation package and module ownership boundaries.
docs/         Architecture, tensor contracts, baseline facts, and decisions.
```

## External baseline boundary

F3Loc is used as a baseline and protocol reference, not copied into this repository. The baseline inventory in [docs/f3loc_baseline_facts.md](docs/f3loc_baseline_facts.md) records the relevant behavior and pinned source revision.

## Data policy

Datasets, floorplan caches, checkpoints, generated figures, external repository clones, secrets, and machine-specific paths are deliberately excluded. Reproduction instructions may name required artifact formats, but do not publish the artifacts themselves.

## Design-first workflow

Use the documents in `docs/` as the source of truth for research design. Record accepted design choices in [docs/decision_log.md](docs/decision_log.md) before implementation changes are made in `track1_core/`.
