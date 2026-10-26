# Experiments versus library

`src/avfploc` is the installed library. It must never import experiments.
Experiments import library components; configs specify a hypothesis, seed,
split, checkpoint and metric definition. Outputs stay outside Git.

`legacy/` is a preservation snapshot of pre-spatial-interaction acoustic and
weighted-fusion experiments. It is not installed and is not a turnkey portable
pipeline: some scripts expect old cache layouts, companion HTML templates,
legacy imports and local launch infrastructure. Original files remain untouched.
Historical failed/invalid-data recipes are provenance, not recommended commands.
For a new run, use the portable package and explicit manifests described in
`docs/hssd_reproduction.md`. Do not present migrated trainer results as identical
to old schedules. Spatial interaction, direct echo-map and case-policy extensions
are intentionally outside the advertised two-branch release.

Future experiment folders should use descriptive names (`fusion_capacity`,
`fusion_supervision`, `acoustic_input_ablation`), never put model implementations
inside experiment-number folders. Legacy import cleanup can proceed one verified
experiment at a time without breaking the historical snapshot.
