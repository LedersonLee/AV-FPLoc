> 역사적 연구 기록. 원본: `acoustic_cues_20261107_1830/PROTOCOL.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Acoustic cue research — active, not complete

User specification: /home/jeongeonlee/.codex/attachments/e8d76840-599b-4374-80d7-92280519a77d/pasted-text-1.txt. Preserve B2, original20support/refined hypotheses, frozen GEOQ and all previous artifacts. No Test observations. Local resources authorized. No new architecture as starting point.

## Hypotheses before new localization outcomes

Htime: ordered2–128ms temporal energy/envelope shape provides within-scene visual ambiguity rescue. Competing explanations: late decay/global spectrum, cross-channel structure, scene-level prior/simulator signature, simple embedding distribution shift.
Hcoherence: current complex adjacent-pair coherency supplies information beyond magnitude/envelope. Coherency is two summaries (early<50ms and full<128ms), not direct TDoA nor known wall paths.
Hspatial: ordered microphone information contributes yaw/pose separation; identical source-to-ring direct distance makes the direct arrival itself an unlikely absolute room-position cue.
Hrepresentation: a cue may be present/decodable yet not useful in final decisions. Negativeprobe/handcraftedfailure does not prove absence.

## Initial stages and gates

1. Catalog every sourcecache/query/reference path; check rawexistence allindexedrecords, metadata/checksums on deterministic16perrole/scene; inspect rendering/sourcegeometry/clock andrawexamplepreprocessing. Prior sampleonly audit not sufficient for all-record availability.
2. Query-only frozenencoder sensitivity screen on31scenes: baseline, removeE/C/N, singleE/C/N, removeimaginarycoherency, E/Ntimeaverage, E/Cfrequencyaverage, channelaverage, earlyEN/lateEN. Explicitly retainC in ENtimewindows, never callthese completeearly/late removal. Query-only screen keeps bank unchanged, hence is OOD sensitivity only, not cue necessity.
3. Candidate-conditioned interpretable descriptors from raw/cache: ordered temporal/spectralenergy, complexcoherence, envelopepeaktiming/shape/decay; waveform direct/early/late/amplitude/GCC probes where verified. Compare identicalfixed20hypotheses and allbankcue-ranking separately. GT onlylabels/evaluation.
4. Train12sceneOOF matchedcapacity cue models /cue-removal/retraining andembedding+cue comparisons. Train-only selection of followuphypotheses, not repeatedVal tuning. Follow promising or anomalousresults with both-query/reference transformations and combinations/restoration controls. Existingstreamablationcheckpoints may be used only aftermatching data,budget,architectureaudit.
5. Report acoustic1/5/20, GT-neighborhoodmass, correct/wrongseparability, fixedsetdiscrimination, frozenfusion/GEOQ, retainedTrainambiguitygroups, visualrescue/harm/scene/seed. Distinguishpure/exchanged andcandidate-reextraction.
6. Representationprobes, cases/signalplots, causalrevision, limitations, reproduction, preservedhashes, requirementaudit. Do not complete atinitialscreen.

Train12 upstreamencoder exposure remains. Val8/Val11 reuseddevelopment. No newindependentscene acquired. Source/simulator are fixed-source-centered active sensing; no claim generalpassivesoundlocalization. No RIRpeak labeledspecificreflectionwithout independentcorrespondence evidence.
