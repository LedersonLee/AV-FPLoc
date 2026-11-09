> 역사적 연구 기록. 원본: `astra_fusion_20261107_0200/fusion_system_audit.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# 융합 시스템 감사 · 2026-11-07

- 범위: 새 학습 이전 읽기 전용 조사 후, 배포 패키지의 동결 UnLoc/E+C+N 점수 융합을 재현한다. 최근 attention 연구 구현은 별도 계열로 기록한다. 모델 발전 단계가 다른 점수를 하나의 방법 효과로 비교하지 않는다.
- 저장소: `${WORKSPACE}/AV-FPLoc`, branch `refactor/hssd-two-branch-fusion`, commit `6ebcf64c22f334542fd623cb3a21005b02a66331`. 상위 track1은 Git 저장소가 아니다. 기존 수정 파일 2개: `scripts/gibson_runtime/global_acoustic_likelihood_gpu.py`, `render_f3loc_echoscan_rirs.py`. 이 작업에서 수정하지 않는다.
- 지시 파일: 홈/상위/작업 폴더에서 `AGENTS.md`를 검색했으나 발견하지 못했다. 연구 지시의 원본은 첨부된 `AV_FPLoc_Astra_Fusion_Research_Goal.md`이다.
- 격리: 앱의 worktree 생성 도구가 실제 Git 저장소를 인식하지 못해 실패했다. 따라서 명시적 대안으로 이 새 실험 폴더만 쓴다. 기존 소스·데이터는 컨테이너에 읽기 전용으로 연결한다. 새 코드 전체와 해시가 변경 내역이며 기존 저장소 diff는 보존한다.
- 자원: RTX4090 24GB, 기존 Python 서버 2개가 합계 약 7GB 점유. 본 실험은 GPU 0GB, CPU 최대 4개, RAM 상한 8GB, 네트워크 없음. `avfploc:cu124-clean` 이미지에서 수행한다. 호스트 python 명령 없음, python3 numpy1.21.5 있음.

## 데이터와 재현 계약

- 부모 split은 HSSD Train116/Val19/Test17, 기존 연구 subset은 Train12/Val3/Test3. 새 선택/학습을 하지 않는다. Train12 명단과 Val3/추가Val5 명단은 `../mbt_remaining_suite_20261106/data/config.json`에 있다. 실행 시 부모 manifest의 Val 포함, Train/Test 비포함을 강제한다. Test 배열은 열지 않는다.
- 이번 초기 진단: Val3의 1704 query. 추가 확인 후보: 과거 개발에 쓰인 다른 Val5의 2486 query. 추가Val은 독립 test가 아니다.
- 입력: `../mbt_remaining_suite_20261106/data/<scene>/{query.npz,fp.npy,xy.npy,windows.npy,receipt.json}`. query/후보는 receipt SHA256으로 검증한다. sample_indices, truth, query 특징은 동일 NPZ에 대응 순서로 저장됐다. GT는 측정에만 사용하고 예측에 전달하지 않는다.
- 후보: 0.2m 간격 XY grid, yaw36개=0..350°/10°. 세계 좌표 X,Z(m), +X=0°/+Z=90°. 후보는 GT에 맞춰 추가/삭제하지 않는다. windows `[C*36,11]`, fp `[C*36,128]`, 출력 `[C,36]`.
- query 입력의 furnished 음향과 구조 평면도 FP 음향의 도메인이 다르다. matched furnished/unfurnished query counterfactual이 이번 cache에 없으므로 가구의 인과 효과는 측정 불가. 원본 RIR SNR/ToA, RGB 밝기·텍스처도 여기에는 없으며 불확실성 proxy만 사용한다.

## 파이프라인

- 시각: `src/avfploc/vision/infer.py`, `geometry/rays.py`. RGB480×640/ImageNet 정규화→UnLoc depth/scale `[Q,40]`. 11개 -50..50° 광선 보간 후 구조 DESDF(0.1m/10m cap)와 Laplace형 오차 점수 `-sum(abs(ray-depth)/scale)/40`.
- 음향: `acoustic/features.py`, `envelope_cache.py`, `model.py`. 48kHz, 원래 direct sample7, 2..128ms, STFT256/hop64, 500..4000Hz, 6mic 순서 유지. Energy CNN/complex coherence CNN/common-removed envelope CNN의 출력 각각128D를 concat384→Linear128+LayerNorm→L2 normalize. Query/FP 공유 encoder. 후보 dot product 점수. Envelope 대역500..1500Hz 및 float16 quantization 계약은 reproduction 문서에 명시.
- 본 cache에는 UnLoc pre-depth `[Q,40,128]`, CNN 토큰 `[Q,52,100]`, mu/scale `[Q,40]`, observed `[Q,128]`가 있다. 이번에 원래 encoder 전체를 재실행하지 않고 검증된 특징에서 점수와 융합을 다시 계산한다. 따라서 baseline 범위는 frozen-feature replay이다.
- 고정 융합: query마다 전체 XY/yaw 점수 z-score, `(1-w)*V+w*A`. w=.1 기존 기준, .5 equal 기준. z-score는 보정된 확률이 아니다.
- 학습 게이트: `fusion/features.py`, `model.py`, `infer.py`. processed58의 GT-free gap/entropy/spread/두 모달리티 합의/점수반응/예측깊이-지도 잔차→Train mean/std→58→128→64→sigmoid weight. GT·scene ID는 inference feature가 아니다. train/val feature 정규화는 Train에서만 fitting.
- 패키지 trainer는 soft-pose CE 또는 margin objective를 지원하며 Val macro joint로 선택한다. 역사적 capacity checkpoint는 다른 scene-balanced40epoch/success target schedule로 학습되었으므로 패키지 trainer 재학습과 동일한 실험이라고 부를 수 없다.

## Attention 판정

- 배포 패키지: scalar score fusion. Q/K/V cross-attention 없음. acoustic projection/concat도 cross-attention이 아니다.
- `../mbt_remaining_suite_20261106/model.py`: modality Transformer + bottleneck latent averaging 교환. 단순 전역 embedding 평균과는 다르지만 직접 cross-modal Q/K/V 호출은 아니다.
- `../direct_attention_stft_20261107/network.py`: 실제 `MultiheadAttention(query=acoustic, key=visual, value=visual)` 구현. CNN52 tokens 또는 complex STFT60 patches를64D로 투영, Vision40×128→64D. 음향 embedding에 bounded residual, FP bank/vision likelihood 고정. 따라서 이 구현은 실제 비대칭 cross-attention이다. 이 구조의 결과를 배포 scalar gate와 혼동하지 않는다.
- 최근 연구의 decoder는 train `(alphaV,alphaA,beta)=(1,1,1)` / inference `(1,.5,.5)`로 다르다(`legacy_model.py`, `model.py`). 별도 연구에서 제어할 수 있는 confound이며 이번 동결 원래 점수에는 적용하지 않는다.

## 체크포인트와 위험

- Vision: `UnLoc/evaluation_checkpoints/epoch020/model.ckpt`, SHA256 `b9626c0f8b5411f365d6151e7a760b8e79fffe5e6ee2abced6caaa7705a1c36f`.
- Acoustic E+C+N: `AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1/runs/all/best.pt`, SHA256 `f7ba7267c66490a034338d19867049e7392fef6795a6b947dfbbdc02f57d27b8`.
- Gate: `ReliabilityFusion/pilots/capacity_probe_v1/runs/128_64_seed42/best.pt`, 기대 SHA256 `b96d7beffb41e20c4e3c79f25ecd612e7029eda1d5bf59db18203c87a827bd28`; 실제 실행 receipt에 digest 기록.
- 위 경로는 `${HSSD_ROOT}` 아래. 체크포인트 목록은 실제 존재 확인 및 release manifest/feature lineage에 근거한다. 모든 대용량 encoder weight를 이번에 새로 로드/재검증한 것은 아니다.
- 이전 `pilots/v1/score_cache`는 E+C의 `2a7f9b...` 모델이고 현재 E+C+N이 아니다. 이 캐시를 현재 baseline으로 쓰지 않는다.
- 과거 Val 재사용, 3scene의 작은 표본, encoder training의 nested OOF 부재, 확률 calibration 미검증, raw observation 품질 지표 부재를 제한사항으로 둔다. modality dropout은 명시적으로 학습/지원되지 않으며 결측 모달리티를 임의의 정상 embedding으로 바꾸지 않는다.

## 명령과 출구 조건

- 조사 명령: `git -C AV-FPLoc status --short`, `branch --show-current`, `rev-parse HEAD`, `worktree list`, `rg --files`, `cat` 관련 README/config/model/train/infer, `nvidia-smi`, `docker ps`, `docker inspect`, Python NPZ shape inspection.
- 실행 명령 원문: [commands.sh](../artifacts.json) (외부 산출물: `commands.sh`). Baseline: `bash experiments/astra_fusion_20261107_0200/commands.sh baseline > experiments/astra_fusion_20261107_0200/baseline.log 2>&1`.
- Phase I 출구 증거: [baseline/receipt.json](../artifacts.json) (외부 산출물: `baseline/receipt.json`), [baseline/metrics.json](../artifacts.json) (외부 산출물: `baseline/metrics.json`), [baseline/samples.json](../artifacts.json) (외부 산출물: `baseline/samples.json`). 로그와 실제 측정이 생성된 후에만 pilot을 시작한다.
- 후속 예산: 최대4개의 고정된 개입(학습 없음) + promising candidate가 있으면1개 추가Val 확인. CPU 예상 각 수분 이내, 총30분 상한. 새 모델/데이터 생성 없음. seed42; deterministic 점수 계산을 seed 반복으로 독립 증거라고 부르지 않는다.
