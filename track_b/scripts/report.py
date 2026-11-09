"""Create small research evidence summaries; never copy predictions/checkpoints."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--docs',type=Path,required=True);args=ap.parse_args();root=args.runs;out=args.docs;out.mkdir(parents=True,exist_ok=True)
    source_commits={'pilot_v1':'55c423833710efe6dd38dbcd58a6e18236ce6f9d','pilot_long_v1':'c268a2f3574e7ba82fe566e39c0803e4f6cf737c',
                    'pilot_rays_v1':subprocess.check_output(['git','rev-parse','e45b121']).decode().strip(),'pilot_zero_v1':subprocess.check_output(['git','rev-parse','e45b121']).decode().strip()}
    summaries={};registry=[];table=[];results=[]
    fields='run_id git_commit timestamp method dataset_version split_hash scene_count train_query_count val_query_count test_query_count input_variant fp_variant grid_xy yaw_bins encoder_regime pretrained_checkpoint_hash parameter_count flops_est gpu seed train_steps gpu_hours best_val_metric test_metric status note'.split()
    for run,commit in source_commits.items():
        if not (root/run/'summary.json').exists():continue
        rr=read(root/run/'summary.json');summaries[run]=rr
        for r in rr:
            row=dict.fromkeys(fields,'NA');row.update(run_id=f"{run}/{r['mode']}/{r['seed']}",git_commit=commit,timestamp='2026-11-09 KST',method=r['mode'],dataset_version='HSSD positive_pairs_v1 furnished',
                split_hash=sha(root/'audit_v1/split_manifest.json'),scene_count=7,train_query_count=64,val_query_count=48,test_query_count=0,input_variant='RGB64x48 + six-channel first8192 samples',
                fp_variant='binary free/unknown + boundary rays' if run=='pilot_rays_v1' else 'binary free/unknown + zero geometry' if run=='pilot_zero_v1' else 'binary free/unknown local patch',
                grid_xy=.5,yaw_bins=12,encoder_regime='joint from scratch',pretrained_checkpoint_hash='none',parameter_count=r['parameter_count'],gpu='RTX4090',seed=r['seed'],train_steps=r['steps'],gpu_hours=r['training_seconds']/3600,
                best_val_metric='not selected: final step fixed',test_metric='not evaluated',status='completed',note='Exploratory; train wall GPUh, not utilization; shared GPU; nll field is soft-target CE')
            registry.append(row)
        for mode in dict.fromkeys(r['mode'] for r in rr):
            rs=[r for r in rr if r['mode']==mode]
            avg=lambda metric,arm='normal':np.mean([r[arm][metric] for r in rs])
            table.append(f"| {run} | {mode} | {len(rs)} | {100*avg('joint'):.2f} | {100*avg('r1'):.2f} | {100*avg('fine'):.2f} | {100*avg('raw20'):.2f} | {100*avg('diverse20'):.2f} | {100*avg('joint','shuffle'):.2f} |")
            results.append({'run':run,'method':mode,'FP-RIR ref required?':'no','New-scene training required?':'no','Raw Coverage@20':avg('raw20'),'Diverse Coverage@20':avg('diverse20'),
                            'MAP@1':avg('joint'),'SupportNMS@1':'NA','Disc@1':'NA','Visual-degraded@1':'NA','Furnished@1':avg('joint'),'NLL':'NA: stored nll is soft-target CE',
                            'new-scene setup s':'NA','storage MB':'NA','warm query ms':'see isolated diagnostic scope','peak VRAM':max(r['peak_vram_bytes'] for r in rs),'train GPUh':sum(r['training_seconds'] for r in rs)/3600})
    for name,rows in [('experiment_registry.csv',registry),('results_master.csv',results)]:
        if rows:
            with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name in ('summary','waveform_sample_qc'):
        if name=='summary':(out/'audit_summary.json').write_text(json.dumps(read(root/'audit_v1/summary.json'),indent=2))
    for filename in ('smoke_v1.json','quantization_corrected.json','stage_identity_audit.json','geometry_diagnostic_v1.json'):
        (out/filename).write_text(json.dumps(read(root/filename),indent=2))
    (out/'pilot_aggregate_results.json').write_text(json.dumps(summaries,indent=2))
    diag=read(root/'diagnostic_v1.json');compact={k:v for k,v in diag.items() if k not in ('rows','io_trace')};compact['io_trace']={k:v for k,v in diag['io_trace'].items() if k!='paths'}
    (out/'diagnostic_summary.json').write_text(json.dumps(compact,indent=2))
    # Paired scene bootstrap: average seeds within each scene first, never treat seeds as scenes.
    comparisons=[]
    specs=[('pilot_v1','joint','pilot_v1','visual'),('pilot_long_v1','joint','pilot_long_v1','visual'),('pilot_rays_v1','joint','pilot_zero_v1','joint'),('pilot_rays_v1','acoustic','pilot_zero_v1','acoustic')]
    for a,am,b,bm in specs:
        if not all((root/r/f'{m}_{seed}.json').exists() for r,m in ((a,am),(b,bm)) for seed in (42,43,44)):continue
        def values(run,mode):
            d={}
            for seed in (42,43,44):
                for row in read(root/run/f'{mode}_{seed}.json')['predictions']:
                    if row['intervention']=='normal':d.setdefault(row['scene'],[]).append(row['joint'])
            return {s:np.mean(v) for s,v in d.items()}
        av,bv=values(a,am),values(b,bm);delta=np.array([av[s]-bv[s] for s in sorted(av)]);rng=np.random.default_rng(42);boots=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        comparisons.append(dict(a=f'{a}/{am}',b=f'{b}/{bm}',scene_count=len(delta),delta_joint_pp=float(delta.mean()*100),ci95_pp=(np.quantile(boots,[.025,.975])*100).tolist(),
                                warning='Exploratory 3-scene resampling interval; no population guarantee or multiple-selection correction'))
    (out/'paired_scene_bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    provenance=dict(code_commits=source_commits,environment_image='avfploc:cu124-clean',image_id='sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318',
       audit_and_smoke_execution='base 5449980 plus then-uncommitted Track B sources subsequently committed as 55c4238; no model checkpoint loaded',
       geometry_diagnostic_execution='base c268a2f plus new ray diagnostic sources subsequently committed as 2555b21',
       files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.json') if p.name not in ('diagnostic_v1.json',) and 'audit_v1' not in str(p)},
       audit_manifest_sha256=sha(root/'audit_v1/manifest.csv'),audit_split_sha256=sha(root/'audit_v1/split_manifest.json'),source_hashes={str(p):sha(p) for p in Path('track_b').rglob('*.py')},
       limitations='Datasets and checkpoints remain external. Docker image tag alone does not guarantee clean-clone reproducibility; use pinned image ID and data hashes.')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    completed={k:len(v) for k,v in summaries.items()};(out/'results.md').write_text('''# Track B 초기 전체 후보 실험 결과

증거 수준: **Exploratory**. 프로토콜 전체 완료가 아닌 Stage 0 및 Stage 1–2 초기 개발 실험이다.

## 관찰

실데이터: 학습 4개 장면×16관측=64, 개발 평가 3개 장면×16관측=48. Seed42/43/44. 아래 수치는 seed 평균이며 독립 장면 수는3이다. Test는 사용하지 않았다. Final step을 사전 고정했으며 best epoch를 선택하지 않았다.

| Run | 모델 | 완료 seed | Joint % | R@1m % | R@0.1m % | Raw@20 % | Diverse@20 % | Audio shuffle Joint % |
|---|---|---|---|---|---|---|---|---|
'''+ '\n'.join(table)+f'\n\n완료 fit 수: `{completed}`. 모든 arm, seed별 결과는 [집계](pilot_aggregate_results.json), [registry](experiment_registry.csv)에 보존한다. '+'''[장면별 paired bootstrap](paired_scene_bootstrap.json)은 seed를 scene으로 세지 않는다. 3개 장면의 좁거나 0인 구간을 일반화 보장으로 해석할 수 없다.

## 해석과 경쟁 설명

120-step 실험은 Train Joint도 약0.5–1%로 낮았다. 합성8관측 암기는 성공했으므로 구현의 기본 미분 경로는 작동하지만 실제 구조 정합 능력은 확인되지 않았다. 저해상도 RGB, 첫170.7ms 음향만 사용, 작은 CNN, 제한된 학습 관측, 좁은 FP 수용영역, 최적화 실패가 모두 경쟁 설명이다. 단순히 audio shuffle에서 점수가 바뀐다는 사실은 유용성 증거가 아니다.

초기 local FP query의 5자리 반올림 특징은 한 장면에서 최대5,128개 후보가 겹쳤다. 반면 16방향 FP boundary ray는 pilot7장면의 후보를 각각 서로 다른 구조 signature로 만들었다. 이는 **표현 분별력 진단**이며 음향 정합 성공이나 성능 향상의 인과 증명이 아니다. 벽/외부/unknown이 혼합된 검은 픽셀을 반향의 물리적 벽으로 단정하지 않았고 RIR reference를 생성하지 않았다.

## 한계와 실패 보존

- 작은 from-scratch feasibility 모델이다. UnLoc/B2+GEOQ의 강한 동조건 재현은 아직 없다. 과거 성능을 비교표에 옮기지 않았다.
- Local visual/acoustic/concat/noninteraction/joint는 allocated parameter수52,324개가 같지만 활성 경로·연산량은 다르다. F3의 엄밀한 capacity-matched 비교로 주장하지 않는다.
- Ray 대 zero는 동일 모듈/초기값/관측 순서/학습 step의 기하 입력 대조다. 추가 FP 정보 효과와 물리적 acoustic mechanism은 구분해야 한다.
- Frozen/partial, variable camera intrinsics, late RIR, 직접파/후기파 개입, 시각 저하, furniture shift, calibration, full0.25m/36학습, 독립 Test, OOF discriminator는 미완료다.
- `nll`로 저장된 필드는 실제로 soft-target cross entropy다. 이 보고서에서는 categorical NLL로 사용하지 않으며 정식 NLL은 NA다.
- 0.5m/12bin pilot full bank의 fine oracle는12.5%; fine0%를 숨기지 않는다. 0.25m/36 audit의 fine upper bound는 별도 보고한다.
- 일부 실행은 같은 GPU에서 겹쳤다. 학습 wall time 및 warm timing은 정확도 검증의 비용 기록이며 방법 간 최종 speed ranking이 아니다.

## 판단 / 다음 반증

Dense retrieval의 낮은 정확도·coverage 때문에 후단 판별기로 진행할 근거가 부족하다. 현재 Stage2 성능 gate는 통과로 선언하지 않는다. 학습 길이만 늘린 결과와 ray-vs-zero 대조를 먼저 확인하고, 실제 Train fit 회복 여부로 최적화 실패와 scene generalization을 분리한다. 구조적으로 구분 가능한 후보도 query-RIR로 구분하지 못하면 acoustic encoder의 시간·위상 표현 및 보조 구조 supervision을 검사한다. H-B1–H-B6는 아직 검증되지 않았다.
''')
    print(json.dumps(completed))

if __name__=='__main__':main()
