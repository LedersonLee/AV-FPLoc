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
