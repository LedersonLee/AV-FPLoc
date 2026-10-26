"""Independent same-query/pose/metric verification; no training or tuning."""
import argparse
import html
from pathlib import Path
import time

import numpy as np

from .contracts import atomic_json, sha
from .data import read

REL='AcousticContrastive/pilots/scene_subset_v1/input_ablation_extended_v1'
ORIGINAL='AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1'
MODES=('energy','coherence','envelope','both','energy_envelope','coherence_envelope','all')
NEW=('coherence','energy_envelope','coherence_envelope')


def verify(root,report):
    root,report=Path(root),Path(report)
    selection=read(root/REL/'selection_frozen.json')
    if selection['test_fitting']:raise ValueError('Test-fitted checkpoint selection')
    if selection['training_summary_sha256']!=sha(root/REL/'summary.json'):
        raise ValueError('Val selection summary changed')
    combined=read(report/'summary.json')
    if combined['status']!='complete' or set(combined['conditions'])!=set(MODES):
        raise ValueError('Seven-condition report incomplete')
    metrics={};arrays={};proofs={};reference=None
    for mode in MODES:
        folder=report/'dense'/mode if mode in NEW else report.parent/'input-ablation-dense'/mode
        records=read(folder/'records.json')
        identity=[(r['scene_id'],r['sample_index'],tuple(r['gt_pose_xzyaw'])) for r in records]
        if len(identity)!=1472 or len(set(identity))!=1472 or len({r[0] for r in identity})!=3:
            raise ValueError('Wrong query coverage')
        if reference is None:reference=identity
        if identity!=reference:raise ValueError('Condition query order/GT differs')
        checkpoint=root/(REL if mode in NEW else ORIGINAL)/'runs'/mode/'best.pt'
        source=read(folder/'summary.json')
        if sha(checkpoint)!=source['acoustic_checkpoint_sha256']:
            raise ValueError('Checkpoint/result lineage mismatch')
        if mode in NEW and selection['models'][mode]!=sha(checkpoint):
            raise ValueError('New checkpoint changed after test started')
        metrics[mode]={};arrays[mode]={}
        for method,expected in combined['conditions'][mode]['micro'].items():
            truth=np.asarray([r['gt_pose_xzyaw'] for r in records])
            pred=np.asarray([r['methods'][method]['pose_xzyaw'] for r in records])
            xy=np.linalg.norm(pred[:,:2]-truth[:,:2],axis=1)
            yaw=abs((pred[:,2]-truth[:,2]+180)%360-180)
            np.testing.assert_allclose(xy,[r['methods'][method]['xy_error_m'] for r in records],atol=1e-8,rtol=0)
            np.testing.assert_allclose(yaw,[r['methods'][method]['yaw_error_deg'] for r in records],atol=1e-8,rtol=0)
            computed={f'recall_{t:g}m':float((xy<=t).mean()) for t in (.1,.5,1)}
            computed['recall_1m_30deg']=float(((xy<=1)&(yaw<=30)).mean())
            for k,v in computed.items():
                if not np.isclose(v,expected[k],atol=1e-12,rtol=0):raise ValueError('Reported metric differs from GT/predictions')
            computed.update({f'yaw_within_{t}deg':float((yaw<=t).mean()) for t in (10,30,50)})
            computed['correct_xy_queries']=int((xy<=1).sum())
            computed['conditional_yaw30_given_xy1m']=float((yaw[xy<=1]<=30).mean()) if np.any(xy<=1) else None
            metrics[mode][method]=computed;arrays[mode][method]=(xy,yaw)
        proofs[mode]=dict(records_sha256=sha(folder/'records.json'),checkpoint_sha256=sha(checkpoint))
    paired={}
    for a,b in [('coherence','energy'),('coherence_envelope','all'),('energy_envelope','all')]:
        ax,ay=arrays[a]['acoustic_only'];bx,by=arrays[b]['acoustic_only']
        valid=(ax<=1)&(bx<=1)
        paired[a+'_vs_'+b]=dict(queries=int(valid.sum()),
            first_yaw30=float((ay[valid]<=30).mean()) if valid.any() else None,
            second_yaw30=float((by[valid]<=30).mean()) if valid.any() else None)
    result=dict(status='verified',queries=1472,scenes=3,conditions=7,metrics=metrics,
        matched_correct_xy_yaw=paired,proofs=proofs,test_fitting=False,updated_unix=time.time())
    atomic_json(root/REL/'independent_verification.json',result)
    atomic_json(report/'independent_verification.json',result)
    rows=[]
    for mode in MODES:
        m=metrics[mode]['acoustic_only'];conditional=m['conditional_yaw30_given_xy1m']
        conditional_text=f'{100*conditional:.2f}' if conditional is not None else '—'
        rows.append('<tr><td>'+mode+'</td>'+''.join(f'<td>{100*m[k]:.2f}</td>' for k in
            ('recall_1m','recall_1m_30deg','yaw_within_30deg'))+
            f'<td>{m["correct_xy_queries"]}</td><td>{conditional_text}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><title>Conditional yaw and integrity check</title>
<style>body{font:16px system-ui;margin:30px;color:#17314d;max-width:1400px}td,th{padding:10px;border:1px solid #ccd6e5}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>
<h1>Global vs conditional yaw · independent metric verification</h1><p><a href="./">Seven-condition comparison</a> · <a href="analysis.html">Analysis</a></p>
<p>Every query identity/GT/checkpoint hash and every XY/yaw error was independently checked against predicted poses.
Same1,472queries /3scenes /7conditions /4policies. No thresholds or weights are fit here.</p>
<table><tr><th>Input</th><th>R1 %</th><th>R1/30° %</th><th>Global yaw≤30° %</th><th>XY-correct queries</th><th>Yaw≤30° given XY-correct %</th></tr>ROWS</table>
<p>Conditional yaw uses different subsets of correct-position queries for different models; it cannot be treated as a controlled causal comparison.
The paired results below restrict BOTH models to the SAME queries where both positions are within1m. Small selected sets still limit generalization.</p>
<pre>PAIRED</pre><p><a href="independent_verification.json">All metrics/checksums</a></p>'''
    (report/'conditional-yaw.html').write_text(page.replace('ROWS',''.join(rows)).replace('PAIRED',html.escape(str(paired))))
    page=report/'index.html';text=page.read_text()
    link='<p><a href="conditional-yaw.html">조건부 방향 성능·독립 검증</a></p>'
    if 'conditional-yaw.html' not in text:
        pending=report/'index.verified.pending.html'
        pending.write_text(text.replace('</html>',link+'</html>'))
        pending.replace(page)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    result=verify(a.root,a.report);print({k:v for k,v in result.items() if k not in ('metrics','proofs')})
