"""Publish honest validation diagnostic tables; not a test benchmark."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from .contracts import atomic_json, sha
from .data import read


def chart(experiments):
    colors=['#2463d4','#bf5b13','#7e3cb5']
    lines=[]
    for color,exp in zip(colors,experiments):
        for key,dash in [('fit',''),('held','stroke-dasharray="6 4"')]:
            points=' '.join(f'{50+r["step"]*2:.1f},{230-r[key]["top1"]*190:.1f}' for r in exp['history'])
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" {dash}/>')
    return '<svg viewBox="0 0 700 270" role="img" aria-label="64-query fit and pose-held-out accuracy"><path d="M50 30V230H660" fill="none" stroke="#888"/><text x="2" y="42">100%</text><text x="20" y="234">0%</text><text x="50" y="255">0</text><text x="610" y="255">300 steps</text>'+''.join(lines)+'</svg>'


def paired_ci(records, baseline):
    base={(r['scene'],r['sample_index']):r for r in baseline}
    groups={}
    for r in records:
        old=base[(r['scene'],r['sample_index'])]
        groups.setdefault((r['scene'],tuple(r['xyz'])),[]).append(r['top1_credit']-old['top1_credit'])
    byscene={s:[np.asarray(v) for (scene,_),v in groups.items() if scene==s] for s,_ in groups}
    rng=np.random.default_rng(42)
    deltas=[]
    for _ in range(2000):
        values=[v[i] for v in byscene.values() for i in rng.integers(0,len(v),len(v))]
        deltas.append(float(np.concatenate(values).mean()))
    return dict(delta_pp=100*float(np.mean([r['top1_credit']-base[(r['scene'],r['sample_index'])]['top1_credit'] for r in records])),
        ci95_pp=(100*np.quantile(deltas,[.025,.975])).tolist(),
        caveat='pose-cluster bootstrap within these 3 val scenes; exploratory, no correction for val model selection')


def run(source, output):
    output.mkdir(parents=True,exist_ok=True)
    diagnostic=read(source/'results.json')
    raw=read(source/'raw_results.json')
    linear=read(source/'linear_results.json')
    checkpoints=read(source/'checkpoint_results.json')
    metric=read(source/'metric_results.json')
    cnn=checkpoints['checkpoints']['best']
    rows=[dict(method='Frozen CNN best (epoch 1)',top1=cnn['sampled']['sampled_pool_top1'])]
    rows.extend(dict(method=name,top1=v['all']) for name,v in raw['splits']['val']['metrics'].items())
    rows.extend(dict(method='Ridge '+name,top1=v['best']['top1']) for name,v in linear['modes'].items())
    best=max(metric['sweep'],key=lambda v:v['val_top1'])
    rows.append(dict(method='57-distance linear metric',top1=best['val_top1']))
    comparisons={name:paired_ci(v['records'],cnn['records']) for name,v in linear['modes'].items()}
    comparisons['pair_metric']=paired_ci(best['records'],cnn['records'])
    summary=dict(scope='Validation diagnostics, NOT test',queries=1704,test_accessed=False,
        rows=rows,comparisons_vs_frozen_CNN=comparisons,
        provenance={p.name:sha(p) for p in source.glob('*results.json')})
    control_path=source/'geometry_control/results.json'
    control=read(control_path) if control_path.exists() else None
    if control is not None:
        summary['geometry_control']=control['summary']
    atomic_json(output/'summary.json',summary)
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','top1'])
        writer.writeheader();writer.writerows(rows)
    table=''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{100*r["top1"]:.2f}%</td></tr>' for r in rows)
    fit=''.join(f'<tr><td>{e["mode"]}</td><td>{100*e["history"][-1]["fit"]["top1"]:.2f}%</td><td>{100*e["history"][-1]["held"]["top1"]:.2f}%</td></tr>' for e in diagnostic['experiments'])
    capture=[('CNN best epoch1',cnn['captured_bank'])]+[('Ridge '+k,v['captured_bank']) for k,v in linear['modes'].items()]
    capturetable=''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{100*v[k]:.2f}%</td>' for k in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg'))+f'<td>{v["median_xy_error_m"]:.2f}m</td></tr>' for name,v in capture)
    per_scene=[]
    for scene in sorted({r['scene'] for r in cnn['records']}):
        a=[r['top1_credit'] for r in cnn['records'] if r['scene']==scene]
        b=[r['top1_credit'] for r in best['records'] if r['scene']==scene]
        per_scene.append(f'<tr><td>{scene}</td><td>{len(a)}</td><td>{100*np.mean(a):.2f}%</td><td>{100*np.mean(b):.2f}%</td></tr>')
    control_html=''
    if control is not None:
        ctr=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["queries"]}</td><td>{100*v["stft_top1"]:.2f}%</td><td>{100*v["coherence_top1"]:.2f}%</td></tr>' for k,v in control['summary'].items())
        control_html=f'<section><h2>5. 形状・再レンダリング 통제 실험</h2><p>각 val scene에서 오류와 무관하게 선택한 8개 위치, 총 24개 쿼리. 모든 조건은 같은 8개 후보 pose/yaw를 사용합니다. fp_repeat: 새 FP 관측→기존 FP 후보. native_stage: native stage 관측→FP 후보. furnished: 가구 포함 관측→FP 후보. native_vs_native: stage 관측→stage 후보. furnished_vs_native: 가구 포함 관측→stage 후보.</p><table><tr><th>조건</th><th>N</th><th>STFT Top-1</th><th>Coherence Top-1</th></tr>{ctr}</table><p>Native stage에는 object로 배치된 문/창문도 없을 수 있으므로, stage와 furnished의 차이를 순수 가구 효과로 단정하면 안 됩니다. FP repeat의 오차는 stochastic renderer와 작은 배열의 yaw 분별 한계도 포함합니다. 소수 통제 쿼리이므로 전체 데이터셋에 일반화한 원인 비율은 아닙니다. <a href="geometry_control.json">쿼리별 통제 결과</a></p></section>'
        atomic_json(output/'geometry_control.json',control)
        normal=control.get('normal_audit',{})
        if normal.get('queries'):
            n=normal['mean']
            control_html += f'<section><h2>면 방향만 반전한 추가 대조군</h2><p>24개 GT 위치에서 geometry 좌표·높이·재질은 유지하고 winding/normal 방향만 반전했습니다. 평균 legacy STFT-L1 거리(낮을수록 유사): native stage↔FP {n["stage_to_outward"]:.2f}→{n["stage_to_inward"]:.2f}, furnished↔FP {n["furnished_to_outward"]:.2f}→{n["furnished_to_inward"]:.2f}. 큰 mismatch가 그대로 남아, 이 표본에서는 면 방향이 주원인이라는 근거가 없습니다. 원본 FP geometry와 RIR은 수정하지 않았습니다. 단일 셰이더/렌더러의 보편적인 양면 처리 동작을 증명하는 테스트는 아닙니다.</p></section>'
        options=[]
        for r in control['rows']:
            if r['condition']=='furnished_repeat':
                key=f'{r["scene"]}_{r["sample_index"]:05d}'
                options.append(f'<option value="{key}">{r["scene"]} / Q{r["sample_index"]:05d}</option>')
        control_html += '<section><h2>통제 쿼리별 RGB·반사 에너지</h2><label>Query <select id="controlQuery">'+''.join(options)+'</select></label><p>곡선은 GT pose의 furnished / FP / native-stage RIR입니다. 직접음 peak+2ms 이후 band energy를 각 RIR의 전체 에너지로 정규화했습니다. 절대 음압 비교가 아니며, 학습 특징의 fixed-clock 규약과 다른 legacy 분석 규약입니다.</p><img id="controlRGB" alt="Selected query RGB" style="width:45%;vertical-align:top"><img id="controlPlot" alt="GT reflection energy in three geometries" style="width:100%"></section><script>const cq=document.getElementById("controlQuery");function updateControl(){document.getElementById("controlRGB").src="control_figures/"+cq.value+"_rgb.png";document.getElementById("controlPlot").src="control_figures/"+cq.value+".png";}cq.addEventListener("change",updateControl);updateControl();</script>'
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HSSD Acoustic learning diagnostics</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;background:#f5f7fb;color:#17243a}}section{{background:white;border:1px solid #d9dfea;border-radius:10px;padding:20px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;max-width:700px}}.warning{{background:#fff0d7;padding:16px}}a{{color:#165db4}}</style>
<h1>HSSD Acoustic 학습 실패 진단</h1><p class="warning">Test 평가가 아닙니다. Train 12 scenes / Val 3 scenes, 1,704 queries. Test 추가 생성·평가는 중단했고 완료 데이터는 보존했습니다. Val로 여러 설정을 비교한 탐색적 결과이며, 독립 test 개선을 주장하지 않습니다.</p>
<section><h2>1. 동일한 8개 후보에서의 정답 선택</h2><p>정답 FP 1개 + 고정 negative 7개. 무작위 선택 12.5%. 좌표·yaw·RGB·scene ID는 모델 입력에 넣지 않았습니다. Ridge 정규화와 metric 규제 강도는 val로 선택했습니다.</p><table><tr><th>방법</th><th>Val Top-1</th></tr>{table}</table></section>
<section><h2>2. 소수 데이터를 학습할 수 있는가?</h2><p>공식 train scene 하나에서 64개 fit / 다른 물리 위치의 64개 held queries. 같은 위치의 yaw는 양쪽에 섞이지 않습니다. 다만 후보 FP RIR 공유까지 금지한 독립 환경 검증은 아닙니다. 최종 step 300 결과입니다.</p><table><tr><th>입력</th><th>Fit</th><th>Pose-held-out</th></tr>{fit}</table>{chart(diagnostic['experiments'])}<p>파랑: energy, 주황: coherence, 보라: both. 실선 fit / 점선 held. 높은 fit 정확도는 구현 전체의 정확성이나 일반화를 보장하지 않습니다.</p></section>
<section><h2>3. 촬영된 전체 pose/yaw 후보에서의 위치 검색</h2><p>각 val scene 내 촬영된 후보 전체에서 검색합니다. 정답이 포함된 captured-pose bank이며 dense grid 성능은 아닙니다. 8후보 정확도 상승이 실제 위치 추정 개선으로 이어지는지 따로 확인합니다.</p><table><tr><th>방법</th><th>R@0.1m</th><th>R@0.5m</th><th>R@1m</th><th>R@1m &amp;30°</th><th>중앙값</th></tr>{capturetable}</table></section>
<section><h2>4. Scene별 pair metric 비교</h2><table><tr><th>Scene</th><th>Queries</th><th>CNN best</th><th>57-distance metric</th></tr>{''.join(per_scene)}</table><p>세 scene 내 물리 위치 단위 paired bootstrap 구간은 summary.json에 있습니다. 모델 선택 편향이나 새 scene에 대한 불확실성을 제거하는 구간은 아닙니다.</p></section>
{control_html}<section><h2>해석과 한계</h2><p>학습 쿼리를 외우는 것은 가능하지만, 같은 scene의 다른 위치부터 일반화가 약합니다. shared CNN만의 문제인지 확인하기 위해 exact legacy STFT, envelope, 낮은 용량의 observed→FP 변환, 시간/주파수별 57개 거리의 가중 학습을 비교했습니다. 성능 상승을 강제하지 않으며, 후보 선택 개선과 위치 추정 개선을 구분합니다. 음향 정보 자체가 무효라거나 visual top-k가 필수라는 결론은 아직 내릴 수 없습니다.</p><p><a href="metrics.csv">CSV</a> · <a href="summary.json">요약·구간·SHA</a> · <a href="results.json">과적합 실험</a> · <a href="raw_results.json">STFT/envelope 쿼리별</a> · <a href="linear_results.json">Ridge</a> · <a href="checkpoint_results.json">CNN best/last</a> · <a href="metric_results.json">Pair metric</a></p></section></html>'''
    for p in source.glob('*results.json'):
        (output/p.name).write_bytes(p.read_bytes())
    text=text.replace('形状・再レンダリング', '형상·재렌더링')
    (output/'index.html').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pause-pilot-page',action='store_true')
    a=p.parse_args();run(a.source,a.output)
    if a.pause_pilot_page:
        parent=a.output.parent
        status=read(parent/'status.json') if (parent/'status.json').exists() else {}
        status.update(status='paused_for_validation_diagnostics',results_available=False,
            diagnostics_available=True,diagnostics_url='diagnostics/',completed_RIRs_preserved=True)
        status.setdefault('test_bank',{})['status']='paused'
        atomic_json(parent/'status.json',status)
        (parent/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>HSSD acoustic diagnostics</title><style>body{font:18px system-ui;max-width:800px;margin:60px auto;padding:20px}</style><h1>HSSD acoustic pilot</h1><p>Validation 일반화 문제 진단을 위해 test 후보 RIR 추가 생성과 test 평가를 중단했습니다. 완료된 RIR과 체크포인트는 모두 보존했습니다.</p><p><a href="diagnostics/">학습·검증 진단 결과: 표 / 통제 실험 / 쿼리별 RGB·반사 곡선</a></p><p>현재 test 네 방식 비교 성능은 산출하지 않았습니다.</p></html>')
