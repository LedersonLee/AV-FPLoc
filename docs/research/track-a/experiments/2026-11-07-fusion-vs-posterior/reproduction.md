# 재현과 원본 근거

원본 연구 폴더: `${WORKSPACE}/experiments/fusion_vs_posterior_20261107_1300`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.

- [원본 재현 문서: reproduce.md](archive/reproduce.md)

## 원본 실행·분석 코드

- `after_preparation.sh`
- `analyze.py`
- `analyze_controls.py`
- `calibration.py`
- `confirm.py`
- `confirmation_controls.py`
- `continue.sh`
- `decoder_labels.py`
- `evaluate_confirmation.py`
- `factorial.py`
- `final_audit.py`
- `independent_prepare.py`
- `interaction_analysis.py`
- `plot.py`
- `report.py`
- `run_script.sh`
- `study.py`
- `supplement.py`
- `verify_confirmation.py`
- `weight_frontier.py`

전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.

## 검증 경계

[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.
