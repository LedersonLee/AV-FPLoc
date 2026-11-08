> 역사적 연구 기록. 원본: `hssd_scaling_20261108_1332/REPRODUCE.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 재현 및 재개

Study root: `${WORKSPACE}/experiments/hssd_scaling_20261108_1332`.
원본 입력/과거 연구는 read-only mount, 현재 study만 `/output` read-write로 실행한다. 기존 checkpoint/캐시를 덮어쓰지 않는다. Test-family를 제외한 고정 scene IDs는 `split_protocol.json`에 있다. Teacher Train116/Val19 노출과 acoustic Val3 선택 이력은 그대로 유지한다.

## 완료 결과 검증

```bash
bash ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/reproduce.sh verify
```

검증은 manifest/파일 목록/해시/최종 결과 개수/원본 dirty patch를 확인한다. Manifest 자체의 해시는 FINAL_VERIFICATION.json에 보존한다. 자기참조 파일과 Python bytecode는 manifest에서 제외한다.

Checkpoint를 다시 실행하는 별도 검사:

```bash
bash ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/reproduce.sh replay
```

같은 GPU·image에서 17개 representative checkpoint의 후보/최종 pose를 재현하고 GT diagnostic key를 제거해도 head 추론이 같음을 확인한다. 원본 입력128query는 전체query 입력/후보와 비교했다. Full query와 pilot은 같은 scene/일부 같은 관측이므로 독립 replication이 아니다.

## 깨끗한 별도 재학습

기존 study를 지우지 않는다. 새 폴더 이름은 반드시 존재하지 않아야 한다.

```bash
python3 ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/bootstrap_replica.py ${WORKSPACE}/experiments/hssd_scaling_replica
bash ${WORKSPACE}/experiments/hssd_scaling_replica/reproduce.sh fit
bash ${WORKSPACE}/experiments/hssd_scaling_replica/reproduce.sh full
```

`fit`는 frozen input extraction → target alignment → full-bank candidate generation → 30 trees/40 neural refits+120 inner → 12 B1/B2 →12 visual refits+36 inner → evidence →94 selected models freeze → Dev19/head-held11 평가를 순차 수행한다. `full`은 기존11scene 전체query에서만 모델을 재평가하며 다시 학습하지 않는다. 중단된 단계는 receipt/complete.json이 있는 결과를 재사용한다. 불완전 산출물이 있으면 해당 로그·receipt 불일치를 먼저 조사하고 조용히 완료로 처리하지 않는다.

필수 기존 경로는 `${WORKSPACE}/experiments/{mbt_remaining_suite_20261106,fusion_vs_posterior_20261107_1300,structural_evidence_20261107_1730,directional_disambiguation_20261108_0123,hypothesis_discrimination_20261107_1538,evidence_refinement_20261028,astra_fusion_v2_20261107_0225}`와 `${WORKSPACE}/AV-FPLoc/src`다. 입력 receipt에는 `/file2` 및 `/ssd1`의 정확한 cache/checkpoint/schedule hash가 있다. 이 자산 없이 공개 데이터만 다운로드해서 즉시 재현되는 패키지라고 주장하지 않는다.

Docker image `avfploc:cu124-clean` (ID `sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`). Host Python은 numpy/scipy/matplotlib/Pillow를 사용한다. `RESOURCE_SUMMARY.json`에 실제 환경 정보를 보존한다. Habitat smoke는 별도 `track1-soundspaces-echoloc:ktx-hssd-20261015` image, `/opt/conda/envs/soundspaces/bin/python`, `/output` workdir, system libstdc++ preload가 필요하며 `render_known_pose.log`에 실행 결과가 있다.

## 분석·그림 재생성

```bash
python3 ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/analyze_results.py
python3 ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/supplementary_analysis.py
python3 ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/analyze_full.py
python3 ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/plot_results.py
python3 ${WORKSPACE}/experiments/hssd_scaling_20261108_1332/build_report.py
```

통계의 bootstrap seed는20261108, draw10,000. 학습seed42/43, scene 내부 시드 평균 뒤 scene-cluster resampling. 단위는 scene11이며 query/seed를 독립 표본으로 세지 않는다. 새 outcome을 보고 training scale/architecture를 선택한 후 독립 평가라고 부르지 않는다.

## 근거에 따른 전체 학습query 후속 실험

`EXPANDED_TRAINING_PROTOCOL.json`은 pilot E0 신호가 전체query에서 약해진 뒤, 유지된 GEOQ 학습량 효과를 시험하기 위해 고정했다. 12full=6,387 /24full=12,627 /24balanced=6,387query. Tree2arms×3scales×2seeds와 neural4arms×3scales×2seeds(각3innerfolds)다. 이미 결과를 확인한11scene은 이제 탐색적 재사용 평가라고 명시한다.

```bash
bash ${WORKSPACE}/experiments/hssd_scaling_replica/reproduce.sh expanded
```

기존12scene의 full input은 원본 해시를 유지한 사본을 만들고 새12scene은 기존 캐시로 encoder만 계산한다. 최초 실행에서는 두 seed 후보 생성을 별도 job으로 병렬화했으며, 재현 wrapper는 순차 실행한다. 동일 각seed 산출물이며 race를 피하기 위해 하나의seed출력에는 한job만 쓴다. `AMBIGUITY_GATE_PROTOCOL.json`은 기존 train entropy threshold 하나를 사용하는 추가 탐색적 policy다. Threshold search 없이 실패도 보존했다.

## OOF·장면별 가중치 복원 대조

전체query `expanded` 단계 뒤 새 replica에서 다음 순서로 실행한다.

```bash
bash ${WORKSPACE}/experiments/hssd_scaling_replica/reproduce.sh oof
bash ${WORKSPACE}/experiments/hssd_scaling_replica/reproduce.sh weighted
```

`oof`는 기존12scene의 실제 B2 제외-fold 출력을12774query-seed 전부 재실행하고, 새12scene은 기존 fullB2(이12scene 미학습)를 유지한다. 같은6arms×3scales×2seeds로36refit+72inner를 수행한다. `weighted`는 같은 OOF 입력의 tree2arms×3scales×2seeds에 장면별 동일 총 가중치를 적용한다. Threshold0가 일차 operating point이며 과거 train-selected seed42=0/seed43=.01도 모든 scale에 고정 적용한다. 새 threshold 탐색은 없다.

원본 GEOQ를12full 조건에서 재구성한 score 최대오차0, 개발19+재사용11의 모든 결정 일치를 검증했다. `recipe_inference_replay.json`은 새48checkpoint 각각의 첫 protocol development/held scene에서 총264condition 출력 재실행을 검증한다. 이 검사에는 GT 항목을 입력하지 않는다.

최종 총478fits=178refits+300innerfits. 분석은 `analyze_oof.py`, `analyze_scene_balanced.py`, `analyze_recipe_controls.py`; 그림은 `plot_oof.py`, `plot_recipe_controls.py`; 최종 보고서는 `build_report.py`다. 결과와 manifest를 보존하려면 재생성은 replica에서 수행한다. 원본 폴더에서 로그·보고서를 변경하면 manifest 검증은 실패해야 정상이다.

OOF-only의 query-weight scaling 이득은 기존 scene-weight를 복원하면 불확실해졌다. 기존 GEOQ를 안정적으로 능가한 새 모델을 선정하지 않았다. 같은 노출·가중치의 score-only보다 candidate geometry가 주는 이득이 일관적이었다. 모든 후속11scene 결과는 탐색적 재사용이며 visual teacher 미노출 검증이 아니다.
