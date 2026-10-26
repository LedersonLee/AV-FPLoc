"""Complete the seven nonempty input subsets using frozen train/val/test splits.

Only C, E+N and C+N are newly trained. Existing E, E+C, N, E+C+N checkpoints and
dense results are hash-pinned/reused. All new best checkpoints freeze before any
new dense test inference. This is exploratory reuse of the original three test
scenes, not a blind final test or a parameter-matched causal ablation.
"""
import argparse
import copy
import fcntl
import html
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .contracts import atomic_json, immutable_json, sha
from .data import read, preflight
from .input_ablation import REL as ORIGINAL, PILOT, evaluate_best_last
from .input_ablation_dense import run as dense_run, publish_comparison
from .train import train, AcousticEncoder

REL = PILOT/'input_ablation_extended_v1'
NEW_MODES = ('coherence','energy_envelope','coherence_envelope')
ALL_MODES = ('energy','coherence','envelope','both','energy_envelope','coherence_envelope','all')
LABELS = dict(energy='STFT energy (E)',coherence='Complex coherence (C)',envelope='Residual envelope (N)',
    both='E + C',energy_envelope='E + N',coherence_envelope='C + N',all='E + C + N')


def publish_val(root, report, summary, status):
    original = read(root/ORIGINAL/'summary.json')
    rows = {r['mode']:r for r in original['runs']}
    rows.update({r['mode']:r for r in summary['runs']})
    table = []
    last_table = []
    for mode in ALL_MODES:
        row = rows.get(mode,{})
        metrics = row.get('best_val_captured_bank',{})
        cells = [LABELS[mode], row.get('status','pending'), row.get('parameters','—'),row.get('best_epoch','—')]
        cells += [f'{100*metrics[k]:.2f}' if k in metrics else '—' for k in
                  ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg')]
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>')
        last = row.get('last_val_captured_bank',{})
        cells = [LABELS[mode],row.get('last_epoch','—')]
        cells += [f'{100*last[k]:.2f}' if k in last else '—' for k in
                  ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg')]
        last_table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HSSD 7-input-subset ablation</title><style>body{font:16px system-ui;margin:30px;max-width:1400px;color:#17314d}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccd6e5}th{background:#eef3fa}.note{padding:18px;background:#fff1d2;line-height:1.7}</style>
<h1>HSSD acoustic: 입력 7조건 비교</h1><p>상태: STATUS</p>
<div class="note">Train12 / 6,387 queries, val3 / 1,704 queries. 아래는 captured-pose VAL bank이며 dense test와 다릅니다.
E=STFT energy(수작업 STFT-L1 점수 아님), C=complex coherency, N=mic-mean-removed envelope.
C, E+N, C+N만 새로 학습하며 기존 네 조건은 그대로 재사용합니다. 전 조건 seed42, batch64, AdamW3e-4, cosine40epochs, patience8,
1positive+7negatives. Best는 val8-candidate loss로만 선택합니다. 파라미터 수가 다르고 single-seed이므로 엄밀한 인과 분해는 아닙니다.</div>
<h2>Val best 비교 (%)</h2><table><tr><th>입력</th><th>상태</th><th>Parameters</th><th>Best epoch</th><th>R.1</th><th>R.5</th><th>R1</th><th>R1/30°</th></tr>ROWS</table>
<h2>Val last 비교 (%)</h2><p>Last는 종료 시점이며, dense test에는 val loss로 선택된 best만 사용합니다.</p>
<table><tr><th>입력</th><th>Last epoch</th><th>R.1</th><th>R.5</th><th>R1</th><th>R1/30°</th></tr>LASTROWS</table>
<p>모든 새 체크포인트를 고정한 후, 동일한 test3 / 1,472 queries / 0.2m·10° dense bank에서 평가합니다.
이전부터 살펴본 test subset이므로 새로운 blind benchmark는 아닙니다.</p>
<p><a href="dense/">추가 3조건 dense test 진행/상세</a> · <a href="analysis.html">입력 제거 효과 분석</a> · <a href="val_summary.json">Val 결과 JSON</a></p></html>'''
    page=page.replace('STATUS',html.escape(status)).replace('LASTROWS',''.join(last_table)).replace('ROWS',''.join(table))
    report.mkdir(parents=True,exist_ok=True)
    (report/'val.html').write_text(page)
    if not (report/'dense_complete.json').exists():(report/'index.html').write_text(page)
    atomic_json(report/'val_summary.json',dict(status=status,runs=[rows[m] for m in ALL_MODES],
        training_summary=summary,selection='VAL eight-candidate loss only'))


def final_analysis(root,vision,report,result):
    """No fitted threshold/weight: paired descriptive results on the same queries."""
    rows=[]; records={};parameters={}
    for mode in ALL_MODES:
        source=(report/'dense'/mode) if mode in NEW_MODES else (
            vision/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/input-ablation-dense'/mode)
        records[mode]=read(source/'records.json')
        summary=read(source/'summary.json')
        expected=summary['acoustic_checkpoint_sha256']
        checkpoint=root/(REL if mode in NEW_MODES else ORIGINAL)/'runs'/mode/'best.pt'
        if sha(checkpoint)!=expected:raise ValueError('Displayed source checkpoint changed')
        for method,metrics in result['conditions'][mode]['micro'].items():
            yaw=np.array([r['methods'][method]['yaw_error_deg'] for r in records[mode]])
            for threshold in (10,30,50):metrics[f'yaw_within_{threshold}deg']=float((yaw<=threshold).mean())
        m=result['conditions'][mode]['micro']['acoustic_only']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        parameters[mode]=sum(p.numel() for p in AcousticEncoder(**state['config']['model']).parameters())
        cells=[LABELS[mode],str(parameters[mode]),*(f'{100*m[k]:.2f}' for k in
            ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','yaw_within_10deg','yaw_within_30deg','yaw_within_50deg'))]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    paired={}
    base=records['all']
    identities=[(r['scene_id'],r['sample_index']) for r in base]
    before=np.array([r['methods']['acoustic_only']['xy_error_m']<=1 and r['methods']['acoustic_only']['yaw_error_deg']<=30 for r in base])
    for mode in ALL_MODES:
        other=records[mode]
        if [(r['scene_id'],r['sample_index']) for r in other]!=identities:raise ValueError('Query pairing changed')
        after=np.array([r['methods']['acoustic_only']['xy_error_m']<=1 and r['methods']['acoustic_only']['yaw_error_deg']<=30 for r in other])
        paired[mode]=dict(repair_vs_all=int((~before&after).sum()),harm_vs_all=int((before&~after).sum()))
    contrasts={}
    for without,label in [('coherence_envelope','Adding STFT energy to C+N'),
                          ('energy_envelope','Adding coherence to E+N'),('both','Adding envelope to E+C')]:
        all_m=result['conditions']['all']['micro']['acoustic_only'];without_m=result['conditions'][without]['micro']['acoustic_only']
        contrasts[without]=dict(label=label,delta_R1_pp=100*(all_m['recall_1m']-without_m['recall_1m']),
            delta_joint_pp=100*(all_m['recall_1m_30deg']-without_m['recall_1m_30deg']))
    body=''.join('<li>'+html.escape(v['label'])+f': R1 {v["delta_R1_pp"]:+.2f} pp; R1/30° {v["delta_joint_pp"]:+.2f} pp.</li>' for v in contrasts.values())
    page='''<!doctype html><meta charset="utf-8"><title>Seven input subsets: analysis</title>
<style>body{font:16px system-ui;margin:30px;max-width:1450px;color:#17314d}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccd6e5}.note{background:#fff1d2;padding:18px;line-height:1.7}pre{white-space:pre-wrap}</style>
<h1>Seven acoustic input subsets: dense test analysis</h1><p><a href="./">네 방식 비교 / RGB·likelihood 상세</a> · <a href="val.html">Val</a></p>
<div class="note">Same original3 test scenes / 1,472 queries. All new models trained from scratch and selected only by val loss.
Unlike frozen-branch neutralization, these models can adapt their remaining encoders and projection.
Single-seed and unequal parameter counts remain confounds. No test-based best-epoch, fusion-weight, K or data filtering.</div>
<h2>Acoustic-only (%)</h2><table><tr><th>Inputs</th><th>Parameters</th><th>R.1</th><th>R.5</th><th>R1</th><th>R1/30°</th><th>Yaw≤10°</th><th>Yaw≤30°</th><th>Yaw≤50°</th></tr>ROWS</table>
<h2>Retrained leave-one-input-out contrasts</h2><ul>CONTRASTS</ul>
<p>Positive delta means adding that input helped in this experiment; it does not establish a universally necessary input.
E retains detailed 0.5–4kHz time-frequency relative energy; N smooths low-band residual timing; C preserves normalized adjacent-mic complex consistency.</p>
<h2>Paired joint successes vs all-input model</h2><pre>PAIRED</pre>
<p><a href="analysis.json">Per-scene metrics, yaw recalls and comparison details</a></p>'''
    page=page.replace('ROWS',''.join(rows)).replace('CONTRASTS',body).replace('PAIRED',html.escape(str(paired)))
    (report/'analysis.html').write_text(page)
    atomic_json(report/'analysis.json',dict(contrasts=contrasts,paired=paired,parameters=parameters,
        conditions=result['conditions'],test_fitting=False,test_scenes=3,queries=1472))


def pipeline(root,vision,report,batch=256):
    root,vision,report=map(Path,(root,vision,report));out=root/REL
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=read(root/ORIGINAL/'summary.json')
    if original['status']!='complete' or original['test_accessed']:raise ValueError('Original training not val-only complete')
    base=read(root/ORIGINAL/'configs/all.json')
    pre=preflight(root,base)
    if pre['status']!='ready':raise ValueError(str(pre['blockers']))
    from .envelope_cache import check
    for split in ('train','val'):check(root,base,split)
    original_dense=root/ORIGINAL/'dense_test/summary.json'
    pinned=read(original_dense)
    contract=dict(original_training_sha256=sha(root/ORIGINAL/'summary.json'),
        original_dense_sha256=sha(original_dense),split_sha256=base['scene_subset_sha256'],
        reused_models={m:sha(root/ORIGINAL/'runs'/m/'best.pt') for m in ('energy','both','envelope','all')},
        new_modes=list(NEW_MODES),test_fitting=False,seed=42,max_epochs=40,patience=8)
    if contract['reused_models']!=pinned['contract']['models']:raise ValueError('Original dense/model identities differ')
    immutable_json(out/'contract.json',contract)
    summary=read(out/'summary.json') if (out/'summary.json').exists() else dict(status='running',test_accessed=False,
        runs=[dict(mode=m,status='pending',parameters=sum(p.numel() for p in AcousticEncoder(features=m).parameters())) for m in NEW_MODES])
    configs={}
    for mode in NEW_MODES:
        cfg=copy.deepcopy(base);cfg['model']['features']=mode
        cfg['wandb'].update(name='subset-input-'+mode,group='input-ablation-seven-v1')
        immutable_json(out/'configs'/f'{mode}.json',cfg);configs[mode]=cfg
    try:
        for i,mode in enumerate(NEW_MODES):
            state='training '+mode
            atomic_json(out/'pipeline_status.json',dict(status='running',stage=state,test_accessed=False,updated_unix=time.time()))
            summary['runs'][i]['status']='training';atomic_json(out/'summary.json',summary)
            publish_val(root,report,summary,state)
            run=out/'runs'/mode
            if not (run/'status.json').exists() or read(run/'status.json')['status']!='complete':
                train(root,configs[mode],run,run/'last.pt' if (run/'last.pt').exists() else None)
            result=evaluate_best_last(root,configs[mode],run)
            summary['runs'][i].update(status='complete',best_epoch=result['best']['epoch'],last_epoch=result['last']['epoch'],
                best_val_loss=read(run/'status.json')['best_val_loss'],best_val_captured_bank=result['best']['metrics'],
                last_val_captured_bank=result['last']['metrics'])
            atomic_json(out/'summary.json',summary);publish_val(root,report,summary,'completed '+mode)
        summary['status']='complete';atomic_json(out/'summary.json',summary)
        # Freeze ALL three new checkpoints before inference on the exploratory test subset.
        immutable_json(out/'selection_frozen.json',dict(models={m:sha(out/'runs'/m/'best.pt') for m in NEW_MODES},
            training_summary_sha256=sha(out/'summary.json'),selection='held-out val8-candidate loss',test_fitting=False))
        publish_val(root,report,summary,'training complete; dense test evaluating')
        atomic_json(out/'pipeline_status.json',dict(status='running',stage='dense_test',test_accessed=True,updated_unix=time.time()))
        dense_run(root,vision,report/'dense',batch,ablation_rel=REL,modes=NEW_MODES,labels=LABELS)
        additional=read(out/'dense_test/summary.json')
        combined=dict(status='complete',conditions={**pinned['conditions'],**additional['conditions']},
            queries=1472,scenes=3,test_fitting=False,selection_frozen_sha256=sha(out/'selection_frozen.json'),
            sources=dict(original=contract['original_dense_sha256'],additional=sha(out/'dense_test/summary.json')))
        final_analysis(root,vision,report,combined)
        links={m:('dense/'+m+'/' if m in NEW_MODES else '../input-ablation-dense/'+m+'/') for m in ALL_MODES}
        publish_comparison(report,combined,modes=ALL_MODES,labels=LABELS,detail_links=links)
        page=report/'index.html';text=page.read_text();page.write_text(text.replace('</html>',
            '<p><a href="analysis.html">입력 제거 효과·방향 성능 분석</a> · <a href="val.html">Val best/last</a></p></html>'))
        atomic_json(report/'dense_complete.json',dict(status='complete'))
        atomic_json(out/'pipeline_status.json',dict(status='complete',stage='complete',test_accessed=True,test_fitting=False,updated_unix=time.time()))
    except Exception:
        atomic_json(out/'pipeline_status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','vision','report'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=256);a=p.parse_args();pipeline(a.root,a.vision,a.report,a.batch_size)
