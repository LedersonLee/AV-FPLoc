> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/reproduce.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# 재현 안내

- 기존 결과를 덮어쓰지 않도록 반드시 새 출력 폴더를 사용한다. 입력 데이터와 체크포인트는 원래 경로에서 읽기 전용으로 사용한다.
- 실행 환경은 `avfploc:cu124-clean`, 원본 checkout은 `${WORKSPACE}`. 의존성 버전은 `dependency.log`, 모델/입력 해시는 각 receipt에 있다.
- 아래 전체 재현은 기존 특징에서 시작한다. 원시 RGB/RIR의 encoder 재추론이나 데이터 생성은 포함하지 않는다.

```bash
export ASTRA_FUSION_OUTPUT_DIR=${WORKSPACE}/experiments/astra_fusion_v2_replay_new
mkdir "$ASTRA_FUSION_OUTPUT_DIR"
cp ${WORKSPACE}/experiments/astra_fusion_v2_20261107_0225/*.py "$ASTRA_FUSION_OUTPUT_DIR/"
cp -a ${WORKSPACE}/experiments/astra_fusion_v2_20261107_0225/deps "$ASTRA_FUSION_OUTPUT_DIR/"
```

실행 도구는 다음 경로의 `run_script.sh`다. 위 환경변수가 지정되면 `/output`만 새 폴더로 바뀐다.

```bash
ASTRA_FUSION_RUNNER=${WORKSPACE}/experiments/astra_fusion_v2_20261107_0225/run_script.sh

# 강한 기준과 실패한 학습형 선택기 재현
bash "$ASTRA_FUSION_RUNNER" study.py prepare --seed 42 --split train_val
bash "$ASTRA_FUSION_RUNNER" study.py fit --kind linear --seed 42
bash "$ASTRA_FUSION_RUNNER" study.py fit --kind tree --seed 42
bash "$ASTRA_FUSION_RUNNER" study.py fit --kind paired --seed 42

# Train에서 선택한 국소 지지도 규칙을 고정해 두 seed에서 확인
bash "$ASTRA_FUSION_RUNNER" support.py --seed 42 --split train
bash "$ASTRA_FUSION_RUNNER" support.py --seed 42 --split val
bash "$ASTRA_FUSION_RUNNER" support.py --seed 42 --split extra_val
mkdir "$ASTRA_FUSION_OUTPUT_DIR/support_seed43"
cp "$ASTRA_FUSION_OUTPUT_DIR/support_seed42/selection.json" "$ASTRA_FUSION_OUTPUT_DIR/support_seed43/selection.json"
bash "$ASTRA_FUSION_RUNNER" support.py --seed 43 --split val
bash "$ASTRA_FUSION_RUNNER" support.py --seed 43 --split extra_val

# 정밀화와 메커니즘 대조. 기존 설정을 그대로 사용한다.
bash "$ASTRA_FUSION_RUNNER" refinement.py --seed 42 --split train
for seed in 42 43; do
  bash "$ASTRA_FUSION_RUNNER" refinement.py --seed "$seed" --split val
  bash "$ASTRA_FUSION_RUNNER" refinement.py --seed "$seed" --split extra_val
  bash "$ASTRA_FUSION_RUNNER" controls.py --seed "$seed"
  bash "$ASTRA_FUSION_RUNNER" final_controls.py --seed "$seed"
done
bash "$ASTRA_FUSION_RUNNER" combine_final.py
bash "$ASTRA_FUSION_RUNNER" final_analysis.py
bash "$ASTRA_FUSION_RUNNER" verify_inference.py
```

- 같은 source/data/environment에서 이 순서로 실행하면 새 결과의 `final_summary.json`을 원본과 비교할 수 있다. GPU/BLAS 차이로 생기는 수치 차이는 별도 확인해야 한다.
- 처음 두 선택기의 원래 소스는 `study_v1.py`에도 보존했다. 현재 `study.py`는 paired 분기만 추가했고 linear/tree 경로는 동일하다.
- 전체 logits를 중복 저장하지 않았다. SHA256으로 고정한 query 특징, FP bank, candidate windows, B2 checkpoint에서 재구성할 수 있다. 최종 pose/오차/GT는 `final_samples.json`에 저장했다.
- Test split 실행 명령은 제공하지 않는다. 이번 승인 범위에서 Test 평가는 수행하지 않았다.

# 새 추론 코드 사용

```python
from inference import SupportFusion

# xy: [C,2] world X,Z in metres
# vision_scores, acoustic_scores: [C,36], same candidates and yaw ordering
fusion = SupportFusion(xy, acoustic_weight=.15, temperature=.05,
                       radius_m=.4, yaw_radius_bins=1, refine=True)
result = fusion.predict(vision_scores, acoustic_scores, normalized=False)
pose_xzyaw = result['pose']
```

- 이미 full-bank z-score된 입력이면 `normalized=True`를 사용한다.
- 이웃 행렬은 scene마다 한 번 구성해 여러 query에서 재사용한다.
- inference에 GT, scene ID, oracle gate는 필요하지 않다. 기존 시각/음향 모델을 대체하지 않는다.
