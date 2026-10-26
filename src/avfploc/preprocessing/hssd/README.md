# HSSD preprocessing

- Purpose: prepare geometry, semantics, RGB, structural targets, and RIR schedules.
- Source: migrated production/research tools; original workspace files remain unchanged.
- Run modules with `python -m avfploc.preprocessing.hssd.<module> --help`.
- [Workflow, dependencies, and safety rules](../../../../docs/hssd_preprocessing.md).

## Modules

| Stage | Entry points |
|---|---|
| Geometry and splits | `build_hssd_unloc_dataset`, `hssd_unloc_dataset`, `splits` |
| Connectivity and curation | `reconcile_hssd_unloc_connectivity`, `build_hssd_unloc_curation_manifest` |
| Semantic maps | `build_hssd_unloc_semantic_maps` |
| RGB and frame export | `render_hssd_unloc_rgb_pilot`, `export_hssd_unloc_frames` |
| Furnished depth QA | `audit_hssd_unloc_furnished_depth`, `reclassify_hssd_furnished_nohit` |
| Replacement views | `replenish_hssd_unloc_furnished_views` |
| Final adapter and validation | `export_hssd_unloc_furnished_adapter`, `validate_hssd_nohit_dataset` |
| Structural targets | `prepare_hssd_unloc_structural_targets` |
| Review HTML | `build_hssd_geometry_audit_review`, `build_hssd_furnished_depth_review` |
| RIR schedules and rendering | `hssd_positive_pairs`, `hssd_negative_rirs`, `hssd_rir_supervisor` |

## Boundaries

- Acoustic features remain in `avfploc.acoustic.features` and `envelope_cache`; do not duplicate them here.
- Native RIR rendering still needs the original renderer directory via `--renderer-dir`.
- Historical geometry audits are diagnostic; use the v4 furnished no-hit policy for release filtering.
- This package is not a one-command, raw-assets-to-release pipeline.
