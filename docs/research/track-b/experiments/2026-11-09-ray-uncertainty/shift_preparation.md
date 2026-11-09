# 사전 고정 열화 조건 준비

진행 중:19Val/1,216질의의 원본 RGB를 중앙 full-height black strip25/50/75%로 변형한 뒤 upstream feature를 재추출한다. 기존 특징의 zeroing이 아니다. 밝기 변형은 포함하지 않고 brightness1을 유지한다. 기존 clean/brightness0.2 캐시는 읽기 전용으로 재사용하고 해당 protocol/hash를 평가 전 검증한다. 세 severity의 어떤 결과도 선택하거나 제외하지 않는다.

Visual 실행은 기존 `cache_foundation_visual`의 SHA14264902902c882f9fd13667c86559910462b758 시점 코드를 사용한다. 25→50→75 순차 handle83993, 각 출력 `features_occlusion<severity>_v1`와 동일명.log. 선택된19Val 전체/64queries,동일 Hypersim ViT-L 체크포인트를 사용한다. 생성 시 RGB sha/query ID를 확인한다. 완료 후 manifest/tensor/기존6Val overlap 검증을 별도로 진행해야 한다.48GiB 여유 공간을 확인하고 시작했으며 모델/기존 캐시를 지우지 않았다.

Audio 코드는 `1c24188091148a53c513e441ed056cc949e5bf3d`. 원본 observed RIR hash/role 및 schedule 경로를 확인하고, 기존 offline canonical alignment와16,384sample 절단/패딩을 적용한다. 먼저 원본 RMS normalize→STFT/IPD→fp16이 기존 spectrum과 bitwise 같아야 진행한다. 그 후6채널 전체 signal RMS 대비noise RMS를0.1로 맞춘 Gaussian 잡음을 더한다. 잡음 seed는 고정seed20261109와scene/query key SHA256으로 만들며, 이 식별자는 offline 변형 결정에만 쓰이고 모델 입력에는 없다. noisy waveform을 기존 RMS/STFT/IPD로 다시 처리한다. 모든 query의 실현 SNR20dB 오차는1e-4dB 이내를 요구한다.

Audio CPU handle86213,출력 `audio_noise20_v1`,로그 동일명.log. 이 변형은 ESS 녹음→deconvolution noise를 모사하지 않는다. 새로운 물성/가구 pair를 생성하지 않으며 Test도 사용하지 않는다. 부분 완료 로그는 전체 조건 검증/성능 결과를 뜻하지 않는다. 현재 학습된 uncertainty 모델의 위치 효과는 NA다.

재현:
```bash
python -m track_b.scripts.cache_foundation_visual --root <HSSD> --cache <scale>/cache_v1 --source <repo>/external/checkouts/UnLoc/depth_anything_v2 --checkpoint <HSSD>/UnLoc/pretrained/depth_anything_v2_metric_hypersim_vitl.pth --out <new-output> --split val --center-occlusion 0.25
python -m track_b.scripts.cache_uncertainty_audio_noise --root <HSSD> --cache <scale>/cache_v1 --out <new-output> --code-sha 1c24188091148a53c513e441ed056cc949e5bf3d
```
첫 명령은0.50/0.75도 각각 새로운 출력 폴더로 수행한다. avfploc:cu124-clean,network none,OMP_NUM_THREADS2,코드/입력 read-only mounts. 원본 RIR/RGB/큰 feature와 spectrum은 Git에 업로드하지 않는다. raw logs도 외부 실험 root에 보존한다.
