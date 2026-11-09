# U0–U4 공통 평가 진행

**실제 smoke 통과, 정상 조건 전체 평가 진행 중. 아직 효과 결론 없음.** Evaluator SHA `bf9a825cf58f584e6803f5026977fc1b32c7aafb`.

동일 frozen means/FP descriptors를 사용해U0는원래squared match,U1–U4는weighted Laplace로 전체 후보를 scoring한다. V/A/fusion 각각 평가하며, 정상 조건에서는 사전 정의한 원래 modality scale 적용의 secondary 진단도 별도로 기록한다. 후보/모델/특징/학습 source를 protocol과hash로 확인한다. GT는 field 생성 후 지표/ray calibration 기록에서만 사용한다.

Smoke는 첫 Val장면의2질의/seed42다.27arm-method-policy×2=54평가 행,16개ray 기록,6개U0 modality/fusion 지표 재현검사를 통과했다([검증](eval_smoke_validation.json), [설정](eval_smoke_protocol.json)). 작은 표본의 수치를 선택 근거로 사용하지 않고, 동일 구현으로19Val×3seed×64의정상 전체 평가를 시작했다. 출력 `eval_normal_v1`,로그같은이름.log,handle98179. 예정98,496행은54,720 primary와43,776secondary로 구분한다. 3,648query evaluations를 독립 scene으로 세지 않는다.

저장값은 Joint/R@1m/fine/strict/raw5·10·20/diverse20/nearest-gridNLL/entropy/confidence/ECE/selective risk/연속오차 및 ray별mean/scale/target/visibility다. Ray target은 오직 사후 진단에 사용한다. 별도 집계에서19-scene paired CI와방향/range/visibility calibration을 계산해야 한다. 현재 timing은 map setup/관측·head/field+ray기록 전송 구간을 나누어 기록하지만, warmup 반복 벤치마크나 end-to-end latency는 아직 아니다.

Audio19장면/1,216질의 생성 완료:clean preprocessing의bitwise 재현 불일치0이며,실현SNR은모두20dB±1e-4dB다.모든생성파일SHA도재확인했다([검증](audio_noise_validation.json),[manifest](audio_noise_manifest.json)). 실제녹음잡음 강건성으로 해석하지 않는다. Visual occlusion3수준은별도완료/overlap검증이남아있다.

재현: `python -m track_b.scripts.evaluate_ray_uncertainty --scale-root <scale> --study-root <study30> --out <new-output> --condition normal --code-sha bf9a825cf58f584e6803f5026977fc1b32c7aafb`; smoke는`--smoke`추가. 기존avfploc:cu124-clean/OMP_NUM_THREADS2/network none/read-only input mounts. 다른조건은같은명령의condition을사전설정명으로변경하며아직실행하지않았다. 평가가끝나기전희망적인부분결과로조건을고르지않는다.

## 열화 입력 검증 완료 및 전체 실행

가림25/50/75%의 각19장면/1,216질의 특징 검증이 완료됐다. 원본 upstream/source/preprocess 일치, feature/input manifest hash, query ID/shape/finite 검사를 통과했고 각 조건의 기존6Val 특징은 bitwise 동일했다. 검증 실행 source `ba040da65e3cf3339a716e9dc8a6af07c1a488d7`. [25%](occlusion25_validation.json), [50%](occlusion50_validation.json), [75%](occlusion75_validation.json).

75% 가림과20dB RIR 잡음 각각 첫 장면2질의/seed42의 실제 평가 smoke가 끝났다. 각각30개 primary 결과와16개 ray 기록을 확인했다([가림](smoke_occlusion75_validation.json), [잡음](smoke_audio_noise20_validation.json)). 정상 U0 replay가 아닌 열화 입력 연결 검증이며, 작은 표본 성능을 조건 선택에 쓰지 않는다.

이후 occlusion25→occlusion50→occlusion75→brightness→audio_shuffle→audio_noise20을 순차 평가하는 실행을 시작했다(handle48041). 각 조건은19Val×3seed×64×5arms×3methods=54,720 primary 행이 예상된다. 출력 `eval_<condition>_v1`, 로그 동일명.log. 실행 source SHA는위ba040da, evaluator 구현은bf9a825에서 변경되지 않았다. 정상98,496행 중43,776개 secondary scale 진단은 열화 primary와 섞지 않는다. 정상 평가(handle98179)와 함께 실행하므로 측정 elapsed는 단독 GPU latency라고 부르지 않는다.

재현은 위 evaluator 명령의 `--condition`과 새 `--out`만 각 사전 지정 조건으로 변경한다. 새 feature verifier는 기존 brightness 기본값을 유지하며, 가림에는 `--brightness 1 --center-occlusion 0.25/0.50/0.75`를 지정하고 기존6Val pilot 캐시를 입력한다. 현재 전체 조건의 효과/CI/negative-transfer 결론은 아직 없다.


정상 조건 전체 평가 완료: 98,496 records, 29,184 ray records, U0 replay 10,944 checks 통과. evaluator source는 `bf9a825cf58f584e6803f5026977fc1b32c7aafb`. 가림 25%는 54,720개 고유 결과 및 유한값 검사를 통과했고, 15개 그룹의 요약 평균과 원시 결과의 재계산 값이 1e-9 이내로 일치했다. 가림 50%는 실행 중이며, 나머지 조건은 같은 순차 작업에서 이어진다.
