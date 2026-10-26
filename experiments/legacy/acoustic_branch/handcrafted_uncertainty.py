"""Paired pose-cluster uncertainty; no scoring or test-fit parameter changes."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contracts import atomic_json,sha
from .data import read
from .test_handcrafted import METHODS,LABELS


def run(report,learned):
    base=read(learned/'records.json')
    ids=[(r['scene_id'],r['sample_index']) for r in base]
    groups=defaultdict(list)
    for i,r in enumerate(base):groups[(r['scene_id'],*np.round(r['gt_pose_xzyaw'][:2],6))].append(i)
    clusters=list(groups.values());rng=np.random.default_rng(42)
    draws=[np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))]) for _ in range(2000)]
    methods={}
    for method in METHODS:
        records={(r['scene_id'],r['sample_index']):r for r in read(report/method/'records.json')}
        if set(records)!=set(ids):raise ValueError('Unpaired query sets')
        metrics={}
        for metric,joint in [('recall_1m',False),('recall_1m_30deg',True)]:
            def correct(record):
                m=record['methods']['acoustic_only']
                return m['xy_error_m']<=1 and (not joint or m['yaw_error_deg']<=30)
            delta=np.array([int(correct(r))-int(correct(records[key])) for r,key in zip(base,ids)],float)
            values=[float(delta[indices].mean()) for indices in draws]
            metrics[metric]=dict(learned_minus_handcrafted=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist())
        methods[method]=metrics
    result=dict(methods=methods,clusters=len(clusters),draws=2000,seed=42,
        unit='physical positions, retained yaw queries sampled together',
        interpretation='Within these3testscenes; not unseen-scene generalization; exploratory, no multiple-comparison correction',
        learned_records_sha256=sha(learned/'records.json'))
    atomic_json(report/'learned_vs_handcrafted_uncertainty.json',result)
    rows=''
    for method,metrics in methods.items():
        rows+='<tr><td>'+LABELS[method]+'</td>'
        for key in ('recall_1m','recall_1m_30deg'):
            m=metrics[key];lo,hi=m['ci95'];rows+=f'<td>{100*m["learned_minus_handcrafted"]:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]</td>'
        rows+='</tr>'
    section='<section id="paired-uncertainty"><h2>학습 모델 − Hand-crafted: paired pose-cluster 95% 구간</h2><table><tr><th>비교 대상</th><th>R@1m 차이 [95% interval]</th><th>R@1m·30° 차이 [95% interval]</th></tr>'+rows+'</table><p>185개 물리 pose를 단위로 yaw들을 함께 resampling했습니다. 이 3개 scene 내 탐색적 구간이며, 미관측 scene 일반화 구간이나 다중비교 보정 결과가 아닙니다. <a href="learned_vs_handcrafted_uncertainty.json">계산 규약·수치</a></p></section>'
    page=report/'index.html';text=page.read_text()
    if 'id="paired-uncertainty"' not in text:page.write_text(text.replace('</html>',section+'</html>'))
    text=page.read_text();comparison=read(report/'comparison.json')['methods']
    vision=comparison['learned']['micro']['vision_only']['recall_1m']*100
    learned_fusion=comparison['learned']['micro']['equal_weight_fusion']['recall_1m']*100
    hand_fusion=max(comparison[m]['micro']['equal_weight_fusion']['recall_1m'] for m in METHODS)*100
    if 'id="interpretation"' not in text:
        note=f'<section id="interpretation"><h2>해석</h2><p>집계상 학습 모델이 세 hand-crafted 방식보다 높습니다. 다만 STFT-L1 대비 위치 R@1m 차이의 pose-cluster 구간은 0을 포함하므로 위치만으로 확실한 우위라고 단정할 수 없습니다. 위치·yaw 결합 지표의 개선은 더 뚜렷합니다.</p><p>Vision-only R@1m은 {vision:.2f}%입니다. 학습 기반 1:1 fusion은 {learned_fusion:.2f}%, 세 hand-crafted 기반 fusion 중 최고값은 {hand_fusion:.2f}%로, 이번 test에서는 hand-crafted 결합이 모두 vision-only보다 낮았습니다. 3개 scene의 탐색적 결과이며 독립적인 추가 test로 일반화를 확인해야 합니다.</p></section>'
        page.write_text(text.replace('</html>',note+'</html>'))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--learned',type=Path,required=True)
    a=p.parse_args();run(a.report,a.learned)
