"""Tables and query inspection for matched native-stage/extruded-FP val results."""
import argparse
import csv
import html
from pathlib import Path
import shutil

import numpy as np

from .contracts import atomic_json, sha
from .data import read


def paired_bootstrap(a,b,key):
    old={(r['scene'],r['sample_index']):r for r in a}
    groups={}
    for r in b:
        previous=old[(r['scene'],r['sample_index'])]
        f=lambda row:row['sampled_credit'] if key=='sampled' else float(row['xy_error_m']<=1)
        groups.setdefault((r['scene'],tuple(r['xyz'])),[]).append(f(r)-f(previous))
    scenes=sorted({s for s,_ in groups})
    clusters=[[np.asarray(v) for (s,_),v in groups.items() if s==scene] for scene in scenes]
    rng=np.random.default_rng(42)
    values=[np.concatenate([v[i] for v in clusters for i in rng.integers(0,len(v),len(v))]).mean() for _ in range(2000)]
    return (100*np.quantile(values,[.025,.975])).tolist()


def run(source,output,rgb_root):
    result=read(source/'results.json');output.mkdir(parents=True,exist_ok=True)
    rows=[];scene_rows=[]
    for geometry,methods in result['methods'].items():
        for method,value in methods.items():
            rows.append(dict(geometry=geometry,method=method,**value['summary']))
            scene_rows.extend(dict(scene=scene,geometry=geometry,method=method,**m) for scene,m in value['per_scene'].items())
    for filename,data in [('metrics.csv',rows),('per_scene.csv',scene_rows)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    comparisons={}
    for method in result['methods']['extruded_FP']:
        a=result['methods']['extruded_FP'][method];b=result['methods']['native_stage'][method]
        comparisons[method]=dict(sampled_delta_pp=100*(b['summary']['sampled_top1']-a['summary']['sampled_top1']),
            sampled_ci95_pp=paired_bootstrap(a['records'],b['records'],'sampled'),
            recall1m_delta_pp=100*(b['summary']['recall_1m']-a['summary']['recall_1m']),
            recall1m_ci95_pp=paired_bootstrap(a['records'],b['records'],'recall'),
            caveat='Physical-position cluster bootstrap within these 3 val scenes; not unseen-scene CI.')
    atomic_json(output/'comparison.json',comparisons)
    shutil.copy2(source/'results.json',output/'results.json')
    table=''.join('<tr><td>'+r['geometry']+'</td><td>'+r['method']+'</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in ('sampled_top1','recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg'))+f'<td>{r["median_xy_error_m"]:.2f}m</td><td>{r["median_yaw_error_deg"]:.0f}°</td></tr>' for r in rows)
    scene_table=''.join(f'<tr><td>{r["scene"]}</td><td>{r["geometry"]}</td><td>{r["method"]}</td><td>{100*r["sampled_top1"]:.2f}%</td><td>{100*r["recall_1m"]:.2f}%</td><td>{100*r["recall_1m_30deg"]:.2f}%</td></tr>' for r in scene_rows)
    # Include all query metrics in JSON; copy RGB lazily is unnecessary for
    # this small diagnostic (1,704 PNGs would duplicate substantial storage).
    first=result['methods']['extruded_FP']['STFT_L1']['records']
    records={f'{r["scene"]}_{r["sample_index"]}':dict(scene=r['scene'],sample_index=r['sample_index'],xyz=r['xyz'],gt_yaw=r['gt_yaw'],values={}) for r in first}
    for g,methods in result['methods'].items():
        for m,data in methods.items():
            for r in data['records']:records[f'{r["scene"]}_{r["sample_index"]}']['values'][g+' / '+m]=r
    atomic_json(output/'queries.json',records)
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3D stage vs extruded FP</title><style>body{{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#f4f7fb;color:#17243a}}section{{background:white;padding:20px;border:1px solid #ddd;border-radius:10px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap}}select{{padding:8px;max-width:100%}}</style><h1>Native 3D stage vs 압출 2DFP — 동일 조건 비교</h1>
<p>Val 3 scenes / 1,704 queries. 동일한 furnished observation RIR, 동일한 후보 XYZ/yaw와 동일한 score를 사용하고 후보 geometry만 바꿨습니다. 학습 모델/vision prior/GT-yaw 고정은 사용하지 않았습니다. 과거와 동일한 native-stage 방식으로 현재 val pose의 RIR을 생성한 matched 비교이지, 서로 다른 과거 test 결과를 병치한 표가 아닙니다.</p>
<section><h2>전체 비교</h2><p>8후보 Top1 = 정답1+고정 negative7 (무작위 12.5%). 나머지 지표 = scene 내 촬영된 모든 pose/yaw 후보에서 검색. 정답이 포함된 captured-pose bank이며 dense grid나 test 성능이 아닙니다.</p><div class="scroll"><table><tr><th>후보 geometry</th><th>방법</th><th>8후보 Top1</th><th>R@0.1m</th><th>R@0.5m</th><th>R@1m</th><th>R@1m &amp;30°</th><th>위치 중앙값</th><th>yaw 중앙값</th></tr>{table}</table></div></section>
<section><h2>Scene별 비교</h2><div class="scroll"><table><tr><th>Scene</th><th>Geometry</th><th>방법</th><th>8후보 Top1</th><th>R@1m</th><th>R@1m &amp;30°</th></tr>{scene_table}</table></div></section>
<section><h2>쿼리별 수치</h2><select id="query"><option>Loading…</option></select><pre id="detail"></pre></section>
<section><h2>조건과 한계</h2><p>STFT_L1은 동료의 frozen band-energy/featurise 경로(48kHz, peak+96 samples, 6144 samples, FFT256/hop64, 0–500/500–1500/1500–4000Hz, sum 정규화)입니다. coherence_L2는 현재 branch의 500–4000Hz 복소 coherence, 2–50/2–128ms 두 구간을 비교합니다. 두 방법의 점수를 서로 직접 비교하거나 융합하지 않습니다.</p><p>Native stage는 원래 mesh를 유지하되 object로 배치된 문/창문도 빠질 수 있습니다. 압출 FP는 내부 개구부를 보존하고 외부를 막으며 단일 천장 높이를 씁니다. 어떤 차이가 성능 변화를 유발했는지까지 분리한 실험은 아닙니다. 마이크 yaw/roll 규약은 양쪽에서 동일합니다. 모든 결과는 미학습 거리 기반 비교입니다.</p><p><a href="metrics.csv">전체 CSV</a> · <a href="per_scene.csv">Scene별 CSV</a> · <a href="comparison.json">Paired delta·bootstrap CI</a> · <a href="results.json">전체 쿼리 결과·규약 SHA</a> · <a href="../diagnostics/">이전 학습 실패 진단</a></p></section>
<script>fetch('queries.json').then(r=>r.json()).then(data=>{{const select=document.getElementById('query');select.replaceChildren();for(const [key,row] of Object.entries(data)){{const o=document.createElement('option');o.value=key;o.textContent=row.scene+' / Q'+row.sample_index;select.append(o)}}function update(){{document.getElementById('detail').textContent=JSON.stringify(data[select.value],null,2)}}select.addEventListener('change',update);update()}})</script></html>'''
    (output/'index.html').write_text(text)
    atomic_json(source/'report.json',dict(status='complete',results_sha256=sha(source/'results.json'),report_dir=str(output)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','rgb-root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.output,a.rgb_root)
