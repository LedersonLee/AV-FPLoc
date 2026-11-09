# Asset notice

This is a noncommercial research diagnostic for AV-FPLoc Track A.

HSSD (Habitat Synthetic Scenes Dataset), Mukul Khanna, Yongsen Mao,
Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg,
Eric Undersander, Angel X. Chang and Manolis Savva (2023), 3DLG-HCVC:
“Habitat Synthetic Scenes Dataset (HSSD-200): An Analysis of 3D Scene Scale
and Realism Tradeoffs for ObjectGoal Navigation.”
https://3dlg-hcvc.github.io/hssd/
https://huggingface.co/datasets/hssd/hssd-scenes

Dataset license: Creative Commons Attribution–NonCommercial 4.0 International:
https://creativecommons.org/licenses/by-nc/4.0/

The selected synthetic RGB observations are existing renders resized to at most
384 × 256 and re-encoded as WebP with metadata stripped. Floorplan raster maps
are resized to at most 1000 × 1000 and contain geometry only. Score fields are
quantized derived model-output display grids. Candidate tables are derived
inference summaries. These modifications do not imply endorsement by HSSD.
No raw RIR, embedding features, model weights, complete RGB dataset, credentials,
private absolute paths, or personal photographs are included.

Interaction and gzip loading conventions reuse the repository owner's
`echoloc/corrected-test/` and `echoloc/input-ablation-seven/` reports. This notice
licenses neither third-party models nor the underlying full dataset anew.
