# Reproduction

Code commit은 provenance/registry 참조. 기존 image ID를 확인한다:
`sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`.
Dataset와 기존 image는 Git에 포함되지 않는다. 입력 hash와 split을 확인해야 하며 clean clone만으로 재현된다고 주장하지 않는다.

실제 실행 shell 구조(repo root에서, 새로운 output디렉터리 사용):

```bash
docker run --rm --network none --gpus 'device=0' \
  --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v "$PWD:/work:ro" \
  -v /file2/jeongeon/AV-FPLoc/datasets/HSSD:/data:ro \
  -v /home/jeongeonlee/track1/experiments/track_b_20261109:/out \
  -w /work --entrypoint python avfploc:cu124-clean \
  -m track_b.scripts.pilot --root /data --audit /out/audit_v1 --out /out/NEW_RUN --steps 2000
```

같은 mount로 다음 명령을 실행했다(이미 존재하는 audit/pilot output은 덮어쓰지 않음):

```bash
python -m pytest -q track_b/tests -p no:cacheprovider
python -m track_b.scripts.audit_dataset --root /data --out /out/audit_v1
python -m track_b.scripts.smoke --out /out/smoke_v1.json
python -m track_b.scripts.pilot --root /data --audit /out/audit_v1 --out /out/pilot_v1 --steps 120
python -m track_b.scripts.pilot --root /data --audit /out/audit_v1 --out /out/pilot_long_v1 --steps 2000
python -m track_b.scripts.diagnose --root /data --pilot /out/pilot_v1 --out /out/diagnostic_v1.json
python -m track_b.scripts.geometry_diagnostic --root /data --pilot /out/pilot_v1 --out /out/geometry_diagnostic_v1.json
python -m track_b.scripts.pilot_geometry --root /data --audit /out/audit_v1 --out /out/pilot_rays_v1 --steps 2000 --geometry rays
```

Ray의 matched control 명령(registry에서 완료 여부 확인):

```bash
python -m track_b.scripts.pilot_geometry --root /data --audit /out/audit_v1 --out /out/pilot_zero_v1 --steps 2000 --geometry zero
```

외부 raw artifacts: `/home/jeongeonlee/track1/experiments/track_b_20261109`.
`audit_v1/{manifest.csv,split_manifest.json,dataset_qc.jsonl,waveform_sample_qc.json}`, run별protocol/checkpoints/per-queryJSON/summary、실패 포함로그 보존. 문서에는 작은 aggregate만 복사한다. 새 명령 실행 전에 같은 scene/query/split/protocol hash를 확인한다.

추가 실행/검증:

```bash
python -m track_b.scripts.profile_full_bank --root /data --audit /out/audit_v1 --out /out/full_bank_profile.json
python -m track_b.scripts.qualitative --root /data --pilot /out/pilot_long_v1 --out /out/qualitative_v1.png
python -m track_b.scripts.failure_analysis --root /data --pilot /out/pilot_rays_v1 --out /out/rays_failure_analysis.json
```

`report.py`는 raw결과의 작은 집계/registry/bootstrap만 Docs로 내보낸다. `figure.py`는 완료된 seed의 집계를 그린다. 실행 환경은 위와 같으며 이 두 도구에는 GPU가 필요하지 않다. `failure_analysis`에서 다른 arm은 해당pilot경로와 새out파일명을 지정한다.
