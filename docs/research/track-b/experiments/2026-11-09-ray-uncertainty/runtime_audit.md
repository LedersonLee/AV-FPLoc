# 실제 checkpoint 추론 경로 점검

**완료: 한 query/seed의15개 full-field 비교 통과. 성능 개선 증거는 아님.** 코드 SHA `757a17af572e3239f5996f5062f7fec6ef819940`. [검증 원본](runtime_validation.json).

첫 Val장면의첫query,seed42 고정 입력으로 U0–U4×V/A/fusion을 검사했다. gt/target/target_hit/query_ids/scene/split key를 제거한 뒤 점수를 계산했다. FP descriptor를 다시 계산한 결과와 cached 결과,chunk2048과1024의전체 field가rtol1e-5/atol2e-5에서 모두 일치했다. 각 비교의 최대 오차, argmax 일치, exact top-score tie 수를 보존했다. 실제 관측 audio를 다음 query로 바꾸거나 scale 방향을 순열하면 fixed field가 달라졌다. 이 변화만으로 acoustic/uncertainty가 유용하다고 결론내리지 않는다.

Warmup 후 U0–U4 추론을 builtins.open/io.open hook으로 추적한 결과 file-open event0건이었다. 사전 checkpoint/관측/FP 캐시 로딩과 native/syscall I/O는 이 검사 범위 밖이다. 따라서 전처리 전체가 GT-free라거나 모든 native 파일 접근이 없다고 주장하지 않는다. 기존 offline body-RIR 정렬에는 schedule 각도 metadata가 쓰인다. 본 scorer는 그런 각도/GT/ID를 받지 않는다.

비용은 각arm warmup2회 후5회 GPU 동기화 시간을 기록했다. map setup은 별도로 측정한다. cached upstream feature 입력부터 local encoders/scale heads/full scoring까지이며 RGB/RIR read·upstream encoder·모델 로딩·후보 추출·평가·map setup은 query 시간에서 제외한다. 다른 GPU 평가가 병행 실행 중인 관측치로, 전용 GPU 평균/scene 전체 대표값/배포 end-to-end latency가 아니다. peak allocated도 해당 프로세스의 현재 cached 자료를 포함한다. 상세 반복값은 JSON을 따른다.

재현: `python -m track_b.scripts.audit_ray_uncertainty_runtime --scale-root <scale> --study-root <study30> --out <new-output> --code-sha 757a17af572e3239f5996f5062f7fec6ef819940`. 기존 avfploc:cu124-clean,OMP_NUM_THREADS2,network none,코드와입력 read-only mounts. exit0. 새 학습/parameter 선택 없이 기존3seed heads 중seed42를 사전 지정했다. 정상/열화 전체 평가와19-scene CI는 별도 진행 중이다.
