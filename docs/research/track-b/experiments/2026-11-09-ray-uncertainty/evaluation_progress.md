# U0–U4 공통 평가 진행

**실제 smoke 통과, 정상 조건 전체 평가 진행 중. 아직 효과 결론 없음.** Evaluator SHA `bf9a825cf58f584e6803f5026977fc1b32c7aafb`.

동일 frozen means/FP descriptors를 사용해U0는원래squared match,U1–U4는weighted Laplace로 전체 후보를 scoring한다. V/A/fusion 각각 평가하며, 정상 조건에서는 사전 정의한 원래 modality scale 적용의 secondary 진단도 별도로 기록한다. 후보/모델/특징/학습 source를 protocol과hash로 확인한다. GT는 field 생성 후 지표/ray calibration 기록에서만 사용한다.

Smoke는 첫 Val장면의2질의/seed42다.27arm-method-policy×2=54평가 행,16개ray 기록,6개U0 modality/fusion 지표 재현검사를 통과했다([검증](eval_smoke_validation.json), [설정](eval_smoke_protocol.json)). 작은 표본의 수치를 선택 근거로 사용하지 않고, 동일 구현으로19Val×3seed×64의정상 전체 평가를 시작했다. 출력 `eval_normal_v1`,로그같은이름.log,handle98179. 예정98,496행은54,720 primary와43,776secondary로 구분한다. 3,648query evaluations를 독립 scene으로 세지 않는다.

저장값은 Joint/R@1m/fine/strict/raw5·10·20/diverse20/nearest-gridNLL/entropy/confidence/ECE/selective risk/연속오차 및 ray별mean/scale/target/visibility다. Ray target은 오직 사후 진단에 사용한다. 별도 집계에서19-scene paired CI와방향/range/visibility calibration을 계산해야 한다. 현재 timing은 map setup/관측·head/field+ray기록 전송 구간을 나누어 기록하지만, warmup 반복 벤치마크나 end-to-end latency는 아직 아니다.

Audio19장면/1,216질의 생성 완료:clean preprocessing의bitwise 재현 불일치0이며,실현SNR은모두20dB±1e-4dB다.모든생성파일SHA도재확인했다([검증](audio_noise_validation.json),[manifest](audio_noise_manifest.json)). 실제녹음잡음 강건성으로 해석하지 않는다. Visual occlusion3수준은별도완료/overlap검증이남아있다.

재현: `python -m track_b.scripts.evaluate_ray_uncertainty --scale-root <scale> --study-root <study30> --out <new-output> --condition normal --code-sha bf9a825cf58f584e6803f5026977fc1b32c7aafb`; smoke는`--smoke`추가. 기존avfploc:cu124-clean/OMP_NUM_THREADS2/network none/read-only input mounts. 다른조건은같은명령의condition을사전설정명으로변경하며아직실행하지않았다. 평가가끝나기전희망적인부분결과로조건을고르지않는다.
