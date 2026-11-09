# Head-only 학습 진행

**smoke와 본 학습3seed 완료.** 실제 source SHA `3caccd76fd46289f9301624255584fe0df490394`.

smoke는 원래 Train 순서 첫 장면의 첫4질의,seed42,2updates다. 관측 특징과 mean은 실제 frozen checkpoint에서 추출했고 원래 predictor forward와 bitwise 동일했다. 기존 expert state 전체/체크포인트 파일은 그대로이며 expert gradient가 없고 scale head만 변경됐다. 양쪽 head는각4,225parameters다. [설정](smoke_protocol.json), [실행 검사](smoke_validation.json). 작은 표본의 loss는 수렴/성능 근거가 아니며 loss에 log density가 포함돼 음수일 수 있다.

이를 통과한 후 Train112×64,seeds42/43/44 각각1000updates,batch32,AdamW1e-3/wd1e-4의 고정 실행을 시작했다. [본 학습 protocol](main_training_protocol.json). CPU에서 장면을 읽어 frozen hidden128/mean/target만 추출해 GPU에 저장한 뒤 작은 head를 학습한다. U1의 modality별 constant는 동일 Train 전체 absolute residual 평균으로 정한다. U2/U4는 추가 fit 없이 동일 U3 head를 재사용한다. Val/Test로 학습·step·temperature를 선택하지 않는다.

외부 `heads_v1`, 로그 `heads_v1.log`, handle2324. 실제 모든 fit 종료와 보존 검사 통과는 다음 확인 대상이며, 현재 완료로 주장하지 않는다. training-only MAE/NLL/coverage는 in-sample diagnostics이고 주요 개발셋 효과 검증과 구분한다. extraction/fit 시간과 allocated peak를 분리 기록하되 peak에는 GPU의 추출 특징이 포함된다. 원본 예측/체크포인트는 Git에 올리지 않는다.

재현: `python -m track_b.scripts.train_ray_uncertainty --scale-root <scale-root> --study-root <study30-root> --out <new-output> --code-sha 3caccd76fd46289f9301624255584fe0df490394`; smoke는 `--smoke` 추가. 동일 avfploc:cu124-clean/network none/read-only input mounts/OMP_NUM_THREADS2를 사용한다. U0 replay comparison과 preflight를 먼저 요구하며 출력은 새 경로다. 기존49개 테스트와 실제 smoke가 선행했다. 실패 fit은 현재 없으며 주요 uncertainty 효과·반증·shift 결과는 아직 미측정이다.

## 완료 확인

프로세스exit0, seeds42/43/44 모두1000steps/7168Train queries, expert state/file bitwise 보존·gradient 없음·head 변경 검사를 통과했다. 저장된3개 head checkpoint SHA256도 재확인했다. [전체 fit 기록](completed_fit_summary.json). Train ray NLL은 V각−1.839/−1.888/−1.858, A−.494/−.508/−.568로 각constant보다 낮았다. 이는 in-sample residual 학습 결과이며 위치성능/열화/generalization 개선은 미검증이다. head optimizer 구간은 시드당약1.21–1.27초이고 별도 feature extraction/기타 I/O 시간이 제외된다.
