"""Chance levels and position-cluster paired uncertainty for planar probes."""
import argparse
from pathlib import Path

import numpy as np

from .contracts import atomic_json
from .data import read,load_manifest


def bootstrap(rows,reference,key):
    previous={(r['scene'],r['sample_index']):r for r in reference}
    groups={}
    value=lambda r:float(r['xy_error_m']<=1 and (key=='xy' or r['yaw_error_deg']<=30))
    for r in rows:
        old=previous[r['scene'],r['sample_index']]
        other=old.get('chance_'+key,value(old) if 'xy_error_m' in old else 0)
        groups.setdefault((r['scene'],tuple(r['xyz'])),[]).append(value(r)-other)
    sums={}
    for (scene,_),v in groups.items():sums.setdefault(scene,[]).append((sum(v),len(v)))
    rng=np.random.default_rng(42)
    numerator=np.zeros(2000);denominator=np.zeros(2000)
    for v in sums.values():
        v=np.asarray(v);indices=rng.integers(0,len(v),(2000,len(v)))
        numerator+=v[indices,0].sum(-1);denominator+=v[indices,1].sum(-1)
    return dict(delta_pp=100*sum(sum(v) for v in groups.values())/len(rows),
        ci95_pp=(100*np.quantile(numerator/denominator,[.025,.975])).tolist())


def run(a):
    data=read(a.source/'results.json')
    queries,_=load_manifest(a.root,'val',read(a.config))
    chance=[]
    for scene in sorted({q['scene_id'] for q in queries}):
        q=[r for r in queries if r['scene_id']==scene]
        xyz=np.asarray([r['observed']['xyz'] for r in q]);xy=xyz[:,[0,2]]
        yaw=np.asarray([r['observed']['yaw'] for r in q])
        d=np.linalg.norm(xy[:,None]-xy[None],axis=-1)
        dy=abs((yaw[:,None]-yaw[None]+180)%360-180)
        for i,r in enumerate(q):chance.append(dict(scene=scene,sample_index=r['sample_index'],xyz=r['observed']['xyz'],
            chance_xy=float((d[i]<=1).mean()),chance_joint=float(((d[i]<=1)&(dy[i]<=30)).mean())))
    output=dict(test_accessed=False,scope='exploratory VAL; no multiplicity correction, not unseen-scene uncertainty',
        chance_recall1m=float(np.mean([r['chance_xy'] for r in chance])),
        chance_joint=float(np.mean([r['chance_joint'] for r in chance])),comparisons={},common_mode_factorial={})
    lines=[]
    baseline=data['methods']['extruded_RLR']['coherence_L2']['records']
    for geometry,methods in data['methods'].items():
        output['comparisons'][geometry]={}
        for method,v in methods.items():
            c=bootstrap(v['records'],chance,'xy');b=bootstrap(v['records'],baseline,'xy')
            output['comparisons'][geometry][method]=dict(vs_chance=c,vs_extruded_coherence=b)
            interval=c['ci95_pp']
            lines.append(f'<tr><td>{geometry}</td><td>{method}</td><td>{100*v["summary"]["recall_1m"]:.2f}%</td><td>{c["delta_pp"]:+.2f}pp</td><td>[{interval[0]:+.2f}, {interval[1]:+.2f}]pp</td></tr>')
        output['common_mode_factorial'][geometry]={
            window:dict(xy=bootstrap(methods[common]['records'],methods[raw]['records'],'xy'),
                        joint=bootstrap(methods[common]['records'],methods[raw]['records'],'joint'))
            for window,raw,common in [('2ms/50ms','coherence_L2','common_coherence_L2'),
                ('.35ms/20ms','early_coherence_L2','early_common_coherence_L2')]}
    atomic_json(a.source/'analysis.json',output);atomic_json(a.report/'analysis.json',output)
    document=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Planar probe uncertainty</title><style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ddd}}</style><h1>무작위 기준·불확실성</h1><p>이 후보 bank의 균등 무작위 선택 기대값: Recall@1m {100*output['chance_recall1m']:.2f}%, Recall@1m·30° {100*output['chance_joint']:.2f}%. 정답 위치가 포함된 bank이므로 0%가 무정보 기준이 아닙니다.</p><p>95% 구간은 같은 scene 내 물리 위치를 cluster로, 모든 yaw를 함께 재표집한 paired bootstrap 2000회입니다. 세 val scene에 조건부인 탐색적 구간이며 여러 설정 비교에 대한 보정·미관측 scene 일반화 보장은 없습니다.</p><table><tr><th>후보 모델</th><th>방법</th><th>R@1m</th><th>무작위 대비</th><th>95% 구간</th></tr>{''.join(lines)}</table><p><a href="index.html">전체 실험·RGB·반사 시간 그래프</a> · <a href="analysis.json">공통성분 제거 factorial 효과·전체 JSON</a></p></html>'''
    (a.report/'uncertainty.html').write_text(document)
    index=a.report/'index.html'
    if index.exists():
        text=index.read_text().replace('<th>GT 이벤트 coverage</th>','<th>원본 timing GT coverage</th>')
        text=text.replace('<section><h2>전체 결과</h2>',
            '<p><a href="uncertainty.html">무작위 기준·위치 단위 bootstrap 분석</a></p><section><h2>전체 결과</h2>',1)
        temp=index.with_name('index.analysis.tmp')
        temp.write_text(text);temp.replace(index)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','source','report'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
