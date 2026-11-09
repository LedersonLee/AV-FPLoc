# M1 실제 후보 생성 경로 검증

상태: development 후보 생성/연결 검증 완료. M1 OOF 학습·판별기 성능은 아직 미완료.

## 질문과 변경

기존 F1 전용 후보 생성기가 M1의 full score field를 직접 사용하면서 같은 posterior에서 후보 support와 특징을 계산할 수 있는가? 구현 오류라면 원래 M1 평가의 MAP/raw20/diverse20 지표와 불일치할 것이다. 단순 feature unit test만으로 실제 checkpoint 연결을 보장할 수 없으므로, 완료된 study21 attention 3개 seed에 대해 실행했다.

`--interaction-run`을 추가했다. scorer의 학습/평가 장면, seed, 완료 step, feature/cache manifest, 모든 precursor checkpoint hash를 검사한다. OOF role은 추가로 scorer/precursor의 동일 fold 명세를 요구한다. 이 검사는 기록의 내부 일관성 확인이며 훈련 전체의 독립 증명은 아니다. 실제 M1 OOF fit은 아직 만들지 않았다.

## 설정·관찰

16Train/6Val,64 queries/scene,3seeds; 기존 full-bank attention checkpoint를 변경 없이 사용했다. 실제 M1 field에서 raw20/diverse20을 뽑고 같은 field를 posterior 특징/support 계산에 전달했다. V/A descriptor와 FP descriptor는 해당 joint model의 학습된 상태에서 얻었다. GT는 후보·특징 생성 뒤 label/metric에만 사용한다.

1,152 query evaluations, 2,304 policy packs가 완료됐다. 각 pack은 중복 없는20 candidate와 유한한20×133 특징을 가진다. 모든 query의 MAP과 정책별 oracle coverage는 원래 attention 결과와 정확히 일치했다(불일치0). [검증](joint_development_validation.json), [요약](joint_development_summary.json), [설정](joint_development_protocol.json), [체크포인트 해시](joint_development_checkpoint_hashes.json).

이 연결 검증에는 새 학습/ablation/독립 Test 평가가 없다. 성능 CI나 일반화 이득은 주장하지 않는다. 기존 Val/upstream 노출은 그대로이며, 이 development 후보를 판별기 학습에 사용하지 않는다. F1에서 학습한 기존 판별기를 M1용 재학습 모델로 부르지 않는다. 다음 필요 조건은 모든 precursor와 interaction을 fold-Train만으로 학습한 OOF M1 후보 생성이다.

## 출처와 재현

실제 builder 코드 SHA `af902d67d38abe42c3149f9bdc36e0b1eac2d76b`. 입력 checkpoint의 학습 SHA는 study21 공식 registry를 따른다(과거 attention42와43/44의 실행 SHA 차이 보존). `avfploc:cu124-clean`, network none, OMP_NUM_THREADS2, GPU4090에서 실행; 원시 pack은 Git에 넣지 않는다. 기존 Track A dirty 두 파일은 변경/사용하지 않았다.

```bash
python -m track_b.scripts.cache_discriminator_candidates --cache <structure>/cache --features <foundation>/features_v1 --visual-run <foundation>/learned_fp_w1000_anchor1000_v1 --acoustic-run <structure>/supervised_v1 --interaction-run <interaction>/foundation_matched_v1 --role development --out <new-output-dir>
```

출력 원본: `/home/jeongeonlee/track1/experiments/track_b_oof_20261109/joint_development_candidates_v1`. 기존 산출물은 보존했다. 본 작업으로 학습 중인 확장 M1의 설정이나 모델은 변경되지 않았다.

## 후속 OOF 학습 준비 — 진행 중

분할 학습 지원 코드 `b26ad3c2ddb72399174442c78a16edd578cb5115`는 `train_foundation_interaction --scene-fold`를 추가한다. 기존 `validate_training_fold`로 원래 Train의 family-disjoint partition임을 확인하고, visual/acoustic precursor의 fold 명세와 정확히 일치해야 한다. 학습/평가 데이터는 이 분할로 제한하며, 입력 캐시 출처와 low-light 원본 데이터 protocol 검사는 유지한다. 분할 인자를 생략한 기존 경로는 유지했다. 전체 테스트44개 통과(기존 경고2개,1.81초). 실제 fold 학습 smoke 검증은 아직 남아 있다.

기존 `folds_v1/fold_0.json`, `fold_1.json` 및 fold별 visual_joint/acoustic precursor를 재사용할 계획이다. 16Train pilot의 OOF M1을 완성하기 위한 작업이며, 현재112Train 확장 M1과 같은 규모의 판별기라고 주장하지 않는다. 8-scene fold scorer와16-scene development scorer의 학습 규모 차이는 기존 study23과 마찬가지로 한계다.

밝기0.2 특징을 원래16Train/6Val 전체에 재추출하는 작업을 시작했다. 실행 SHA는 위 코드 SHA, 외부 출력 `track_b_oof_20261109/features_lowlight_all_v1`, 로그는 같은 root의 `features_lowlight_all_v1.log`, 실행 handle85566. 기존 raw observations를 읽고 frozen Hypersim ViT-L을 사용하며 Test는 읽지 않는다. 전체 완료/해시/기존6Val overlap 검증 전에는 fold 학습을 시작하지 않는다. 실행 중 상태이며 새로운 성능 결과가 아니다.

재현: 기존 `cache_foundation_visual` 명령에서 `--cache <structure>/cache --brightness .2 --out <new-output-dir>`를 사용하고 split 기본값all을 유지한다. 원본 HSSD 경로/업스트림 source/checkpoint는 study19/본 작업의 기존 설정과 동일하다. CPU 보관 방식을 사용한2-step fold smoke 이후, attention에 한해2fold×3seed×1000step의 정해진 학습 및 OOF 후보 생성을 진행한다. 후단 비교는 기존 MAP/support/score-only/geometry를 유지하고 development 후보를 판별기 학습에 넣지 않는다.

## OOF 실행 전 검증 완료와 본 학습 시작

밝기 특징 추출은22개 장면/1,408개 질의 모두 완료했고, 파일/입력 manifest hash, query IDs, 유한한 tensor shape를 검증했다. 기존6Val 특징은 bitwise 동일했다([검증](joint_lowlight_validation.json)). 검증 코드 `884864a35590bb259b34fd8c2e4e785fae9b16ca`는 기존19Val 검증의 기본 동작을 유지하며 `--split all --expected-scenes 22`를 추가로 지원한다.

음성 대조에서는 전체16Train precursor를 fold0에 주면 `Precursor fold mismatch`로 거부하고 protocol/체크포인트를 생성하지 않았다([기록](joint_precursor_negative_validation.json)). 예상 실패 traceback은 외부 `joint_wrong_precursor_negative_v1/expected_failure.txt`에 보존했다.

올바른 fold0의8Train/8held-fold, attention42, CPU scene residency,2-step smoke가 완료됐다. 1,536 평가 행(8×64×3arms)과 두 번째 update에서 여섯 branch의 양수 gradient를 확인했다([설정](joint_fold_smoke_protocol.json), [검증](joint_fold_smoke_validation.json)). 이는 수렴/일반화 성공 증거가 아니다.

이후 attention의2fold×3seed×1000step 본 학습을 순차 실행하기 시작했다. 출력 `joint_fold0_v1`, `joint_fold1_v1`; 로그 같은 이름.log; live handle79859. 실행 SHA `884864a35590bb259b34fd8c2e4e785fae9b16ca`. 밝기/aux/anchor/optimizer 설정은 smoke protocol과 같고 steps1000/seeds42,43,44로만 확장한다. source cohort는 기존16Train pilot; 원래6Val는 OOF 모델 학습·fold 평가에 사용하지 않는다. 주 확장 실험과 GPU를 공유하므로 학습 시간은 전용 GPU 효율 비교 근거가 아니다. 최종6fit/OOF 후보/판별기 비교는 아직 진행 중이다.

재현은 `train_foundation_interaction`에 `--scene-fold <oof>/folds_v1/fold_<k>.json --visual-run <oof>/fold<k>_visual_joint_v1 --acoustic-run <oof>/fold<k>_acoustic_v1 --cache <structure>/cache --visual-cache <foundation>/features_v1 --lowlight-cache <oof>/features_lowlight_all_v1 --scene-residency cpu --steps 1000 --seeds 42 43 44 --modes attention --code-sha 884864a35590bb259b34fd8c2e4e785fae9b16ca --out <new-output>`를 사용하며 k=0,1을 순차 실행한다. 출력은 새 폴더여야 한다.

## 후단 입력 모델 일관성 검사

`523d0130e820f6695f243afafa8a1e246eb7f6f6`에서 판별기 학습 CLI에 `--fold-candidates <fold0> <fold1>` 및 `--development-candidates <dev>`를 추가했다. 기존 root 기반 기본 경로는 유지한다. 각 입력의 scorer와 upstream feature protocol SHA가 같고 두 fold의 학습 집합이 상대 held-fold 집합과 같아야 한다. 실행 protocol에는 실제 scorer를 기록해 M1 실행을 F1 backend로 잘못 표시하지 않는다. 기존 pilot의 질의 수1024/384 제약은 유지하므로 확장 규모에 자동 적용되는 코드는 아니다.

음성 대조: 기존 F1 OOF 후보와 M1 development 후보를 섞으면 `Candidate scorer families differ`로 학습/출력 폴더 생성 전에 거부했다([검증](joint_mixed_scorer_validation.json)). 원래 F1 세 protocol은 새 scorer/source/complement 검사에 모두 통과했다. 전체 판별기를 재학습한 검증은 아니며, 실제 M1 OOF 후보가 완료된 뒤 정해진12개 classifier fit에서 긍정 경로를 검증한다. 예상 실패 원문은 외부 `mixed_scorer_guard_validation_v1/expected_failure.txt`에 보존했다.
