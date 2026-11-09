> 역사적 연구 기록. 원본: `hypothesis_discrimination_20261107_1538/PROTOCOL.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Hypothesis extraction and discrimination

Local resources freely authorized. Prior artifacts and existing services preserved; Test sealed. Val8 and Val11 are both reused development sets. No fresh-data generalization claim is permitted. Previous goal turn completed its empirical cycle; this is new research, not a restart of that experiment.

Primary target: actual Top-1 improvement over fixed fusion + support/refinement, not oracle-only coverage. B2 epoch8 scores/candidate bank unchanged. Begin seed42 Train12 OOF and Val8; extend seed43 and reused Val11 only after useful diagnostics. Model/threshold selection uses Train scene folds only. Upstream B2 OOF and ranker CV are not fully nested, so Train CV is a selection diagnostic.

Extraction comparisons: full-score NMS, Euclidean-support NMS, connectivity-aware-support NMS (20centers each). Same strict prediction/evaluation. Each center is refined by maximum original fusion score within its extractor's0.4m/±10deg neighborhood. Preserve center and refined pose separately; evaluate coverage/diversity of both. SE2 suppression uses1m/30deg and no GT. Same ranker architecture/features across extractors; evaluate extracted-first baseline, learned re-ranking, and oracle separately.

Connectivity graph: original candidate cells on0.2m lattice; adjacent8-neighbor edges permitted only if densely sampled segment is free in original1cm floorplan raster. Shortest-path neighborhoods bounded at0.4m and1m. This is a discretized floorplan-geodesic approximation, not Habitat navmesh; missing-grid and path-length discretization can confound wall effects, so record degree/support changes and compare controls. Verify every candidate projects to free space; no GT used in graph.

Features: per-hypothesis visual/audio/fused scores, gaps, local distributions, modality agreement, Euclidean/graph support and moments, uncertainty from original predicted depth. No GT, scene IDs or absolute coordinates in model input. Labels only indicate final hypothesis success. Keep feature manifest and ablations. Runtime excludes encoder/score matching when reporting extraction/ranking cost; include per-scene graph construction separately.

Train ranker: initial gradient-boosted candidate-success classifier, scene-balanced sample weights;4scene-fold validation and threshold search including no-switch baseline. Failure motivates revised ranking objective/features/extraction, not automatic termination. Hypothesis-level attention remains an option if supported by evidence. New validation assets will be investigated after first viable ranking improvement; do not call already used scenes independent.
