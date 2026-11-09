# 재현

외부 root `/home/jeongeonlee/track1/experiments/track_b_uncertainty_20261109`. 입력 root는 protocol/input_validation에 명시한 Study27 root. 원본 checkpoint, 캐시, 대형 평가 파일은 Git에 넣지 않는다.

```bash
python3 -m track_b.scripts.audit_uncertainty_inputs --root <scale-root> --out <new-audit-dir> --code-sha 96db20f075ded8f5c66234496c7b2dfbfb3741cf
python -m track_b.scripts.independent_fusion --cache <scale-root>/cache_v1 --run <scale-root>/scaling_fits_v1/n112_budget1_acoustic_attempt1 --visual-run <scale-root>/scaling_fits_v1/n112_budget1_visual_joint_attempt1 --visual-cache <scale-root>/features_v1 --out <new-replay-dir> --fixed-only --extended-metrics
```

감사는 host Python3 stdlib, GPU replay는 avfploc:cu124-clean, OMP_NUM_THREADS2, PYTHONPATH=/work, 입력/코드 read-only mounts, network none. 출력 `preflight_v1` 완료, `u0_replay_v1` 완료. 원래6개 model 학습 source는5cf803681c2d144f3a620bc00c1ab6b841e48e7c이며 각 checksum은input_validation에 기록했다. 새 uncertainty 학습 source는 `3caccd76fd46289f9301624255584fe0df490394`이다. 초기 preflight 이후 3개 seed 학습이 완료됐으며 checkpoint 해시는 completed_fit_summary.json에 기록했다.


## 실제 학습 및 평가 재현

아래 명령은 컨테이너 내부 경로를 사용한다. `/work`는 저장소의 읽기 전용 mount, `/scale`은 `/home/jeongeonlee/track1/experiments/track_b_scale_20261109`의 읽기 전용 mount, `/results`는 Study30 외부 root다. `avfploc:cu124-clean` (PyTorch 2.4.1+cu124), GPU 사용, `PYTHONPATH=/work`, `OMP_NUM_THREADS=2`, `--network none`에서 실행했다. 모든 출력 디렉터리는 새 이름이어야 하며 기존 산출물을 덮어쓰지 않는다. 아래 `reproduce_*` 경로는 새 실행용 예시이며 이미 실행한 원본 경로가 아니다.

```bash
python -m track_b.scripts.train_ray_uncertainty --scale-root /scale --study-root /results --out /results/reproduce_heads --code-sha 3caccd76fd46289f9301624255584fe0df490394
python -m track_b.scripts.evaluate_ray_uncertainty --scale-root /scale --study-root /results --out /results/reproduce_normal --condition normal --code-sha bf9a825cf58f584e6803f5026977fc1b32c7aafb
```

원본 학습 출력은 `heads_v1`이다. 평가기는 `heads_v1`을 명시적으로 읽으므로 위 평가 재현은 보존된 원본 head를 사용한다. 새로 학습한 head까지 평가하려면 별도 Study root에 동일한 protocol·preflight·replay·열화 입력을 준비하고 새 head를 그 root의 `heads_v1`에 생성한다. 원본 `heads_v1`을 교체하지 않는다. Train 입력과 사전 감사/replay 산출물이 없는 clean clone만으로는 재현되지 않는다.

열화 평가의 실제 실행 source는 `ba040da65e3cf3339a716e9dc8a6af07c1a488d7`이며 evaluator 내용은 정상 평가 source와 같다. 같은 평가 명령에서 `--condition`을 `occlusion25`, `occlusion50`, `occlusion75`, `brightness`, `audio_shuffle`, `audio_noise20`으로 각각 바꾸고 출력도 서로 다른 새 경로를 지정한다. 원본 출력은 `eval_<condition>_v1`이다. evaluator는 original RGB에서 재추출한 `features_occlusion25_v1`, `features_occlusion50_v1`, `features_occlusion75_v1`, scale root의 `features_lowlight_v1`, body-RIR에서 변환한 `audio_noise20_v1`을 사용한다. 각 입력 manifest·protocol·검증 결과는 이 문서 폴더와 외부 root에 보존했다.

모든 7개 조건의 원본 평가가 완료된 뒤에만 다음 집계를 실행할 수 있다. 이 명령은 준비된 재현 명령이며, 전체 조건이 끝나기 전에는 실행 완료를 뜻하지 않는다.

```bash
python -m track_b.scripts.summarize_ray_uncertainty --root /results --out /results/aggregate_v1 --code-sha 8f2454a5ce50f0fff059dcc180da2d07057c1920
```

위 집계 source의 전체 SHA는 provenance.json의 `aggregator_sha`를 따른다. 19개 장면을 재표집하며 seed/query를 장면 수로 세지 않는다. 최종 결과와 저장된 평가 요약의 일치 검사, 정상 대비 열화 paired 변화, 고정 scale-policy 보조 비교를 포함한다. 실행 이후의 검증 결과는 별도 기록한다.


## 최종 집계 완료

최종 산출물은 외부 root의 `aggregate_v2`이다. 앞선 `aggregate_v1`은 code_sha 인자 오기입 때문에 provenance 정정을 위해 보존 후 대체했다. 동일 source `8f2454a5ce50f0fff059dcc180da2d07057c1920`로 `--out /results/aggregate_v2`를 지정해 완료했고 수치 출력은 v1과 bytes 단위로 같다. 모든 평가 조건의 실행이 종료됐고117개 그룹과 원시 결과 재계산 검사가 통과했다. 재현 시에는 다른 새 출력 디렉터리를 사용한다.
