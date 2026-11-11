# Reproducibility, execution provenance, and cost

## 실제 실행 코드

| 단계 | Commit SHA | 비고 |
|---|---|---|
| 첫1-ring pilot | base `a444ab5e9d6f56ba98ff4d78650b2c1bdd2279df` + dirty source | 소스 SHA256은 manifest. 이후6a6fc1e의 renderer와 동일; pilot을 덮어쓰지 않고 primary에 포함 |
| 나머지35 controlled rings | `6a6fc1e83fa8f060d773393988e3da4b535b65a7` | 36rings 전체는 동일 renderer source |
| Finite paths, HSSD preparation, controlled analysis | `2c33fe3a9abaef3e70f953004d639d3169f7e97c` | 실제 finite specular paths |
| HSSD 6controls 최초 | `4643d58d841acc5c7e93a20a1a1d4a889ef1972b` | random bounds correction 전; 로컬v1 보존 |
| HSSD 6controls 최종 | `fdbfb08d37b7a6b98deb5ecae58ffaa74438b4cc` | primary matched 결과 불변, 모든 random windows 공통 support 내 |
| Timing diagnostics42rings | `2f2993046e11e65644da02b935688351ff5a1d96` | samplerate / rays / depth / diffraction |
| Mesh diagnostics12rings | `d3edb972b3a3bed0004c93c074b49d1f88971ad7` | closed box / one-sided |
| Direct-path diagnostics6rings | `770cc6941d40c6bd92961c891a0e3481c585b97b` | 같은 길이의 직접 경로 |
| Subgroups/timing 최종 | `42ee981c63e3de24c6a95664fb17a2ab9ea7d62c` | circular bin boundary correction |
| 독립 검산 | `38599bf9e903a4796f2f0557123c5f474abe115c` | geometry, yaw, source/candidate integrity, scalar power replay |
| 최종 보고서/그림 생성 | `afbf80b741d6c50782bd0684e5a0c8edeeb09991` | 최종 Docs commit과 구분 |

[Source/input hashes](reproducibility_manifest.json)는384개 실제 waveform SHA256,12개 candidate/FP/architectural-slice/schedule hashes, renderer helpers와실행 소스 해시를 보존한다. 코드/문서가 같은 저장소에 있으나 실제 실험 SHA를 문서 commit으로 대체하지 않았다.

## 입력·모델 의존성과 환경

- Dataset root: `${HSSD_ROOT}` = `/file2/jeongeon/AV-FPLoc/datasets/HSSD`. 학습/추론을 새로 하지 않았다.
- 실제 pure-visual Top20: `${WORKSPACE}/experiments/directional_structure_20261109/candidates/<scene>/pure_visual.npz`. `indices`, `pred`, `query_ids`만 경로 생성에 사용한다. Candidate pose는 bank XY와index yaw로 독립 재구성해12장면 모두 일치했다. 파일에 함께 저장된 GT labels/음향점수는 사용하지 않았다.
- Raw lookup: 같은 study의 `signals/<scene>/receipt.json`, 원본 Train acquisition schedule. 기존 visual teacher 및 bank의 upstream 학습 노출을 상속한다. 현재 실행에서 model checkpoint를 load/fit하지 않으므로 새 checkpoint SHA는 N/A다. 기존 후보 제작 과정은 Step0 링크의 prepare_candidates/reproduction에 기록되어 있다.
- Architectural audit: `${WORKSPACE}/experiments/acoustic_geometry_20261110/targets_v3/<scene>.npz` 중 `segments_1.5`만 읽는다. 기존 target depth/pose labels를 window selection에 사용하지 않는다.
- FP: `${HSSD_ROOT}/RIR/Geometry/<scene>/floorplan.glb`, nativeY-up. External renderer helper는 `${WORKSPACE}/3Dto2D_FP/scripts/render_hssd_paired_rirs.py`; 해시는 manifest에 포함했다.
- Renderer image `track1-soundspaces-echoloc:latest`, 실제 image ID `sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1`; Habitat0.2.2, Python `/opt/conda/envs/soundspaces/bin/python`. Tag만으로 같은 환경을 보장하지 않는다.
- Host Python3.10.12, NumPy1.21.5, SciPy1.8.0, Pillow9.0.1, Matplotlib3.5.1. 별도 study-local dependency trimesh3.23.5 (`deps/`, global환경 변경 없음).
- RLR RNG를 통제하는 공개 seed binding이 없으므로 fresh render는 bitwise 재현을 주장하지 않는다. 저장된96rings와설정·hash를 보존했다. 분석은저장 RIR에대해결정론적이다.

**Clean clone만으로 재현되지 않는다.** 위 private/local HSSD 자산, upstream cached visual candidates, architectural slices, renderer image와외부helper가 필요하다. Raw data·weights·캐시·전체 로그는 Git에 넣지 않았다. HTML에는12개 대표 case의480bin min/max display envelope와선택 기하만 포함되며 원본6ch RIR는 로컬에 남겼다.

## 재현 명령

원본 산출물 보호를 위해 새 출력 폴더를 사용한다. 아래 `WORKSPACE`는 기존 프로젝트와데이터가있는 동일host 경로이며 현재 구현은 일부 historical input 경로를 명시적으로 사용한다. 다른host에서는prepare/analyze의입력경로를설정해주고manifest hashes를확인해야 한다.

```bash
WORKSPACE=/home/jeongeonlee/track1
REPO="$WORKSPACE/AV-FPLoc"
STUDY="$WORKSPACE/experiments/reflection_observability_20261111_replay"
CODE="$REPO/scripts/research/track_a/reflection_observability_20261111"
mkdir -p "$STUDY"
# Host에pip가없어격리된기존Python컨테이너로study-local dependency만설치했다.
docker run --rm --cpus 2 --user "$(id -u):$(id -g)" \
  -v "$STUDY:/output" avfploc:cu124-clean python -m pip install \
  --target /output/deps --no-deps trimesh==3.23.5
# 함수는같은mount/resource설정으로선택한renderer를실행한다.
render_job() {
  local renderer_file="$1"
  shift
  docker run --rm --network none --cpus 4 --memory 8g \
    --user "$(id -u):$(id -g)" \
    --entrypoint /opt/conda/envs/soundspaces/bin/python \
    -e CUDA_VISIBLE_DEVICES=-1 -e EGL_PLATFORM=surfaceless \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    -v "$WORKSPACE:/workspace:ro" -v "$STUDY:/output:rw" -w /output \
    track1-soundspaces-echoloc:latest \
    "/workspace/AV-FPLoc/scripts/research/track_a/reflection_observability_20261111/$renderer_file" "$@"
}
render_job render_controlled.py --out /output/controlled --limit 1
# 위pilot처리율·저장량을확인한뒤나머지실행(existing receipt는skip).
render_job render_controlled.py --out /output/controlled
render_job render_diagnostic.py --out /output
OPENBLAS_NUM_THREADS=2 python3 "$CODE/analyze_controlled.py" --root "$STUDY"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/prepare_hssd.py" --out "$STUDY"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/analyze_hssd.py" --root "$STUDY"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/diagnose_results.py" --root "$STUDY"
python3 "$CODE/build_viewer.py" --root "$STUDY" --out "$STUDY/interactive.html"
# Standalone browser self-test;원본RIR를외부서버에전송하지않는다.
python3 - "$STUDY" <<'PYQA'
import json, re, subprocess, sys
from pathlib import Path
p=Path(sys.argv[1])
r=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',
 '--allow-file-access-from-files','--dump-dom',
 (p/'interactive.html').as_uri()+'?selftest=1'],capture_output=True,text=True,timeout=60)
n=re.search(r'data-tests="(\d+)"',r.stdout)
assert r.returncode==0 and n and int(n[1])==69
(p/'viewer_verification.json').write_text(json.dumps(dict(browser='Chrome headless',
 candidate_states_rendered=69,ready='data-ready="true"' in r.stdout,exit_code=0)))
PYQA
python3 "$CODE/build_reports.py" --root "$STUDY" --docs "$STUDY/report"
```

기존보존실험의검산명령은다음과같다. `verify.py`의v1대조는이실험의보존v1파일을요구하므로새clone의처음실행에는그부분을그대로적용할수없다.

```bash
OPENBLAS_NUM_THREADS=2 python3 "$CODE/verify.py" \
  --root "$WORKSPACE/experiments/reflection_observability_20261111"
python3 "$REPO/docs/research/tools/validate_archive.py"
```

Browser QA: standalone HTML을 Chrome headless로 열고 `?selftest=1`에서69개의 case/candidate 상태를 모두 렌더링한다. Body `data-ready=true`, `data-tests=69`와exit0을검사했다. `?case=9&candidate=19`로HSSD Top20 예시를직접열수있다. Controlled/HSSD 화면과evidence.png를직접확인했다.

## 계산·저장 비용

Pilot1ring은0.756s/751472bytes. 초기36ring 추정≈27s/27MB였으며 실제36ring의합계9.384s/56.915MB로복도·room tail때문에저장이더컸다. 추가60ring은10.573s. 합계96개 신규6ch RIR; 이시간은각renderer receipt의scene initialization/render구간합이며 컨테이너시작·설치·분석·전체대화시간은제외한다.

HSSD geometry준비33.015s,첫6control분석92.330s,random-bound 수정후최종73.110s. 하위군·문서·브라우저단계는별도 wallclock 계측을하지않아 전체end-to-end compute cost를위합계로과장하지않는다. Host20logicalCPU에서렌더컨테이너4CPU/8GiB,acousticthreads2로제한했고허용80%보다낮다. Host분석OPENBLAS2threads;GPU/Node2/유료자원은사용하지않았다. 공유host이므로isolated benchmark는아니다.

로컬study폴더는실패로그·dependencies·보존v1·96raw RIR를포함한다. Git문서는약3.3MB이며원본RIR/weights/caches는제외했다. Protocol 변경과실패조건은원본log/JSON을보존했고결과보고서에서정정근거를연결했다.

## 남은 기술적 한계

Reflection-path GT와 합성 전 energy histogram,wall IDs,backendtravel times를현재binding에서얻지못했다. 실제timing mismatch의내부원인규명에는해당export또는검증가능한renderer소스가필요하다. 물리예상시간에맞는first-order검출기가검증됐다고주장하지않으며,실험완료는양성결론과동의어가아니다.
