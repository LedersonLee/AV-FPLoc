# Environment / Dataset audit

## A. 실제 환경과 데이터

- Workspace: `/home/jeongeonlee/track1/AV-FPLoc`; branch `refactor/hssd-two-branch-fusion`; origin은 프로젝트 설정과 일치. 시작 base19c1bd0, 문서 migration이 별도 Commit5449980으로 진행됨.
- RTX4090 24,564MiB, 시작 시 기존 Python2개가 약7,040MiB 사용. 중단하지 않았다. RAM62GiB 중 available약50GiB, workspace여유65GiB, SSD여유2TiB, NFS여유11TiB.
- Host기본 `python`없음; aibot환경 torch없음. 기존 Docker `avfploc:cu124-clean`에서 PyTorch2.4.1+cu124, NumPy1.24.4, SciPy1.10.1 확인. Network disabled, dataset read-only, 현재 UID/GID로 실행.
- Habitat-local import는 habitat_sim0.3.3 성공. SoundSpaces acoustic 렌더 실행/버전 검증은 미실시. 신규 RIR 생성 없음. 과거 sidecar의 CPU-only acoustic 설정과 waveform hash는 확인했으나 과거 simulator binary 동일성은 미확인.
- Dataset root `/file2/jeongeon/AV-FPLoc/datasets/HSSD`; SSD mirror는 디렉터리 존재만 확인, 동등한 최신 복제라고 가정하지 않음.
- Official split116Train/19Val/17Test. Train61,502 및 Val9,965, 총71,467query의 RGB·Observed RIR 존재 및 캡처 pose/yaw 정합 확인. Missing0, alignment failure0, footprint exclusion0. Full dataset 디스크 용량은 미측정; 표본132waveform 합85,778,328bytes만 실제 측정했다.
- 132개 non-Test-family 장면에서 observed RIR1개씩 확인: 6채널·finite·SHA·centre 모두 통과. 792개 채널의 first32sample peak는 모두7. 전체71,467query 파형을 검증한 것이 아니다.
- RGB640×480, HFoV106.2602°, 높이1.5m. RIR48kHz, 6ch radius.05m, colocated Tx와 array centre. 원본 microphone angles `[π,4π/3,5π/3,2π,π/3,2π/3]`; world offsets `(r cos a,0,-r sin a)`.
- Logical yaw는 +X→+Z, Habitat rotation은 `-yaw-90°`. FLU right-handed world는 `(X,-Z)`, yaw부호 반전. Camera forward·mic channel roll을 analytic test로 확인했다. 실제 hardware calibration 주장이 아니다.

## B. 누수·호환성 위험

Test는 scene ID와 stage identity metadata만 확인했다. Test RGB·RIR·pose·score를 읽지 않았다. Source scene-prefix family를 보수적으로 묶어 Test-family Train3개와 Val-family Train1개를 제거했으며 development112Train/19Val가 남았다. 전체 split의 stage template 동일 ID 중복0, Train/Val byte-identical FP중복0. **서로 다른 mesh의 near-duplicate geometry까지 배제한 것은 아니다.**

Pilot은 미리 정한 SHA256(scene ID) 순서의 Train4/Val3과 균일query16개를 사용한다. 기존 HSSD 연구에서 Val19가 재사용되었으므로 scratch encoder라도 pristine independent 평가라고 부르지 않는다. 기존 visual teacher나 acoustic checkpoint는 이번 모델에 사용하지 않았다.

FP는1cm raster,255free/0blocked, +Xright/+Zdown. Black은 wall/exterior/unknown혼합이다. 학습 입력은 free/unknown으로 구분하며 물리 wall truth가 아니다. Candidate mask는 전체 FP만 사용하고 GT trajectory로 component를 선택하지 않았다. 기존 RIR/FP파일은 schema에 존재하지만 모델/loader에서 읽지 않는다. Frozen checkpoint 진단의 Python audit hook으로 observed reads112, reference reads0 확인; OS전체 syscall trace는 아니다.

## C. 실제 구현 및 메모리 정책

- `data/geometry.py`: `candidate_bank`, `world_to_pixel`, `canonical_rir`, `mic_offsets`, `rigid_transform`.
- `data/rays.py`: `boundary_rays(free,spacing,origin,xy,...)`; FP boundary만 계산. Waveform/reference 예측 없음.
- `models/dense.py`: `DenseScorer.forward(rgb,rir,floorplan,calibration,candidate_grid,chunk_size=512)`, `dense_loss`, `topk`, `rasterize`.
- `models/ray_dense.py`: 동일 scorer에 FP16ray range+16hit flag를 조건화. zero-geometry arm은 같은 모듈·parameter수.
- `scripts/audit_dataset.py`, `pilot.py`, `pilot_geometry.py`, `diagnose.py`, `geometry_diagnostic.py`, `smoke.py`는 실제 실행/검증 가능한 CLI다.
- 0.25m/36bin audit 후보 수18,072–375,156/scene. Pilot0.5m/12bin에서는3,576–15,192후보를 **전부** 평가했다. 후보 pruning/sampled negatives 없음. Chunk512, float32, full-bank softmax. FP features는0.1m에서 계산한다.
- 이 작은 pilot에서 GPU peak allocated는 집계JSON에 기록. 0.25m/36/fullRGB학습 메모리 상한은 측정하지 않았으므로 해당 규모의 실행 가능성을 주장하지 않는다.

## D. 실행된 최소 검증

초기 geometry4개→dense9개→mask10개→rays12개→geometry control포함13개, modality isolation포함14개 pytest통과. 전체 bank indexing, invalid `-inf` scatter, GT없는 forward signature, chunk/full equivalence, encoder/interaction gradient, frozen encoder gradient없음, audio shuffle score변화, 새 FP크기, map-frame rigid transform, analytic camera/mic calibration, FP boundary censoring을 확인했다. 임의 GT를 바꾸는 dataset-level leakage test, 모든 file access의 OS tracing, 완전한 profiler test는 추가 검증이 필요하다.

Synthetic8query/168candidate/301evaluation steps: CE5.124→0.064, exactMAP0→100%. 이는 시스템 암기 smoke이고 localization evidence가 아니다.

0.25m grid GT footprint inclusion100%, 최근접 grid최대거리0.141421m. Fine≤0.1m oracle는 numerical tolerance1e-6m를 적용하면 micro52.498%, scene macro52.607%이다. 최초 strict floating 비교의 macro48.474%는 경계값 오차로 과소계산되어 [정정 기록](quantization_corrected.json)을 보존했다. 학습/평가 변경에 GT를 쓰지 않았고 mask를 보정하지 않았다.

## E. Gate 판정

Stage0 analytic coordinate/mic checks 및 데이터 metadata정합은 통과. Full waveform QA, near-duplicate geometry, camera horizontal ray photometric calibration는 미완료. 제한을 명시한 development pilot을 수행했으며 Gate0/1 전체 최종 certification으로 확대 해석하지 않는다. Stage2의 유용한 joint improvement는 아직 미확인, 후단 discriminator 진행 근거 없음. 다음 실제 명령은 [재현 문서](reproduction.md)에 있다.

## Full-bank 실제 forward 검증

가장 큰 eligible 개발 장면103997940_171031257에서0.25m/36bin **375,156개 후보**의 실제 forward와 `[1,128,206,36]` masked field를 검증했다. 모두 finite valid scores이고 invalid은−∞. Random initialization으로 정확도를 평가하지 않았다. Warm3회 후5회 평균1471.78ms, peak allocated59,366,400bytes이며 동시 GPU작업 영향이 있다. I/O·전처리·Top20 제외, 해당 해상도의 full training 미검증. [측정](full_bank_profile.json).
