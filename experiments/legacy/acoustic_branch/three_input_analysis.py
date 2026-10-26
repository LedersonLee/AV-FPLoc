"""Train/val-only capacity control and frozen-branch intervention diagnostics.

No test data is accessed. Branch removal is an out-of-distribution intervention,
not a retrained ablation. Its neutral activation is estimated on train only.
"""
import argparse
import copy
import fcntl
import html
from pathlib import Path
import time

import numpy as np
import torch

from .contracts import atomic_json, immutable_json, sha
from .data import read, load_manifest, check_cache
from .envelope_cache import check as envelope_check
from .train import AcousticEncoder, FeatureStore, train, encode_refs

PILOT = 'AcousticContrastive/pilots/scene_subset_v1'
BASE = PILOT+'/input_ablation_v1'
REL = PILOT+'/three_input_analysis_v1'


def metrics(xy, yaw):
    xy, yaw = np.asarray(xy), np.asarray(yaw)
    result = {f'recall_{t:g}m':float((xy<=t).mean()) for t in (.1,.5,1)}
    result['recall_1m_30deg'] = float(((xy<=1)&(yaw<=30)).mean())
    result.update({f'yaw_within_{t}deg':float((yaw<=t).mean()) for t in (10,30,50)})
    return result


def grouped_comparison(reference, comparison):
    if [(r['scene'],r['sample_index']) for r in reference] != [(r['scene'],r['sample_index']) for r in comparison]:
        raise ValueError('Paired query identities differ')
    before = np.array([r['xy']<=1 and r['yaw']<=30 for r in reference])
    after = np.array([r['xy']<=1 and r['yaw']<=30 for r in comparison])
    return dict(repair=int((~before&after).sum()), harm=int((before&~after).sum()),
        both_correct=int((before&after).sum()), both_fail=int((~before&~after).sum()))


@torch.no_grad()
def evaluate(model, store, queries, batch=128):
    model.eval()
    rows = []
    for scene in sorted({q['scene_id'] for q in queries}):
        group = [q for q in queries if q['scene_id']==scene]
        refs = [q['candidates'][0] for q in group]
        fp = encode_refs(model,store,refs,torch.device('cuda'),batch).numpy()
        observed = encode_refs(model,store,[q['observed'] for q in group],torch.device('cuda'),batch).numpy()
        scores = observed@fp.T
        winner = scores.argmax(-1)
        xyz = np.array([r['xyz'] for r in refs])
        yaw = np.array([r['yaw'] for r in refs])
        xy_error = np.linalg.norm((xyz[winner]-xyz)[:,[0,2]],axis=1)
        yaw_error = abs((yaw[winner]-yaw+180)%360-180)
        for i,q in enumerate(group):
            spatial_wrong = np.linalg.norm((xyz-xyz[i])[:,[0,2]],axis=1)>1
            yaw_wrong = (np.linalg.norm((xyz-xyz[i])[:,[0,2]],axis=1)<1e-6)&(abs((yaw-yaw[i]+180)%360-180)>30)
            rows.append(dict(scene=scene,sample_index=q['sample_index'],xy=float(xy_error[i]),yaw=float(yaw_error[i]),
                spatial_gt_margin=float(scores[i,i]-scores[i,spatial_wrong].max()),
                same_xy_yaw_gt_margin=float(scores[i,i]-scores[i,yaw_wrong].max())))
    return dict(metrics=metrics([r['xy'] for r in rows],[r['yaw'] for r in rows]),
        scenes={s:metrics([r['xy'] for r in rows if r['scene']==s],
                          [r['yaw'] for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})},
        rows=rows)


def publish(root, report, summary):
    report.mkdir(parents=True,exist_ok=True)
    atomic_json(root/REL/'summary.json',summary)
    atomic_json(report/'summary.json',summary)
    table=[]
    for name,result in summary.get('conditions',{}).items():
        m=result['metrics']
        table.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{100*m[k]:.2f}</td>' for k in
            ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','yaw_within_30deg'))+'</tr>')
    pairs=[]
    for name,row in summary.get('paired',{}).items():
        pairs.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{row[k]}</td>' for k in
            ('repair','harm','both_correct','both_fail'))+'</tr>')
    scene_rows=[]
    for name,result in summary.get('conditions',{}).items():
        for scene,m in result['scenes'].items():
            scene_rows.append('<tr><td>'+html.escape(name)+'</td><td>'+html.escape(scene)+'</td>'+
                f'<td>{100*m["recall_1m"]:.2f}</td><td>{100*m["recall_1m_30deg"]:.2f}</td></tr>')
    conditions=summary.get('conditions',{})
    findings='Capacity control still running; do not attribute all gains to input complementarity yet.'
    if 'both_width40_trained' in conditions:
        full=conditions['all_intact']['metrics'];wide=conditions['both_width40_trained']['metrics']
        findings=(f'Three inputs exceed the near-capacity-matched E+C control by '
            f'{100*(full["recall_1m"]-wide["recall_1m"]):.2f} pp at R1 and '
            f'{100*(full["recall_1m_30deg"]-wide["recall_1m_30deg"]):.2f} pp at R1/30°. '
            'Thus parameter count alone does not explain this single-seed result. '
            'Coherency intervention primarily damages heading; energy/envelope interventions also damage position. '
            'These are complementary-cue indications, not proof of clutter invariance or a unique causal mechanism.')
    text='''<!doctype html><meta charset="utf-8"><title>Three acoustic inputs: mechanism audit</title>
<style>body{font:16px system-ui;max-width:1300px;margin:32px;color:#172a42}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccd6e3}.note{background:#fff1cc;padding:16px;line-height:1.6}pre{white-space:pre-wrap}</style>
<h1>Why three inputs? Controlled train/val audit</h1><p>Status: STATUS</p>
<div class="note">Train12 / val3 only. No test arrays, labels or checkpoint selection.
All recalls below use the captured-pose validation bank, not the dense test grid.
Branch interventions replace one branch's 128-D output by its train-only mean; the other branches and all model weights stay fixed.
This is an OOD intervention, not proof of causal importance or a retrained ablation.</div>
<h2>Interpretation</h2><p>FINDINGS</p>
<h2>Validation results (%)</h2><table><tr><th>Condition</th><th>R.1</th><th>R.5</th><th>R1</th><th>R1/30°</th><th>Yaw≤30°</th></tr>ROWS</table>
<h2>Capacity control</h2><p>Energy+coherence width32: 544,160 parameters. Three-input width32: 703,232.
Energy+coherence width40: 696,808 (0.91% below three-input). Same frozen pairs, seed42, global batch64, optimizer, schedule and val-loss selection.
Wider topology is not an exactly parameter-matched architecture. Single-seed, three-scene evidence remains exploratory.</p>
<h2>What the inputs preserve</h2><ul><li>STFT energy: ordered-mic reflection strength across time/frequency; no phase.</li>
<li>Complex coherency: adjacent-mic normalized relative phase and consistency, but no absolute amplitude.</li>
<li>Mean-removed envelope: low-frequency reflection timing/shape after suppressing instantaneous shared microphone content; no carrier phase.
Shared content is not necessarily ceiling-only; clutter can remain.</li></ul>
<h2>Paired joint-success changes</h2><p>For neutral_* rows: before=all_intact, after=branch intervention.
For *_to_all: before=that separately trained model, after=all_intact. Same 1,704 queries.</p>
<table><tr><th>Comparison</th><th>Repair</th><th>Harm</th><th>Both correct</th><th>Both fail</th></tr>PAIRS</table>
<h2>Per-scene check</h2><table><tr><th>Condition</th><th>Val scene</th><th>R1</th><th>R1/30°</th></tr>SCENES</table>
<p><a href="summary.json">Complete per-query errors and GT margins</a></p>'''
    text=text.replace('STATUS',html.escape(summary['status'])).replace('FINDINGS',html.escape(findings)).replace('ROWS',''.join(table)).replace('PAIRS',''.join(pairs)).replace('SCENES',''.join(scene_rows))
    pending=report/'index.pending.html';pending.write_text(text);pending.replace(report/'index.html')


def run(root,report):
    root=Path(root);out=root/REL;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'analysis.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    config=read(root/BASE/'configs/all.json')
    contract=dict(config_sha256=sha(root/BASE/'configs/all.json'),
        checkpoint_sha256=sha(root/BASE/'runs/all/best.pt'),test_accessed=False,
        capacity_control='E+C width40 vs all width32; same training protocol',
        intervention='train-mean branch activation replacement; no test data')
    immutable_json(out/'contract.json',contract)
    summary=dict(status='evaluating frozen branches',conditions={},paired={},test_accessed=False,contract=contract)
    publish(root,report,summary)
    stores={};queries={}
    for split in ('train','val'):
        queries[split],digest=load_manifest(root,split,config)
        locations,_=check_cache(root,config,split,digest,full=False)
        stores[split]=FeatureStore(locations,envelope_locations=envelope_check(root,config,split)[0])
    try:
        checkpoint=torch.load(root/BASE/'runs/all/best.pt',map_location='cpu',weights_only=False)
        model=AcousticEncoder(**config['model']).cuda().eval();model.load_state_dict(checkpoint['model'])
        branches=('energy','coherence','envelope')
        sums={name:torch.zeros(128,dtype=torch.float64,device='cuda') for name in branches}
        counts={name:0 for name in branches}
        def collect(name):
            def hook(module,inputs,output):
                sums[name].add_(output.detach().double().sum(0));counts[name]+=len(output)
            return hook
        hooks=[getattr(model,n).register_forward_hook(collect(n)) for n in branches]
        rng=np.random.default_rng(42)
        selected=rng.choice(len(queries['train']),min(2048,len(queries['train'])),replace=False)
        refs=[r for i in selected for r in (queries['train'][i]['observed'],queries['train'][i]['candidates'][0])]
        encode_refs(model,stores['train'],refs,torch.device('cuda'),128)
        for hook in hooks:hook.remove()
        means={n:(sums[n]/counts[n]).float() for n in branches}
        atomic_json(out/'train_activation_means.json',dict(queries=len(selected),
            means={n:v.cpu().tolist() for n,v in means.items()},counts=counts,test_accessed=False))
        summary['conditions']['all_intact']=evaluate(model,stores['val'],queries['val'])
        publish(root,report,summary)
        for name in branches:
            hook=getattr(model,name).register_forward_hook(lambda module,inputs,output,m=means[name]:m.expand_as(output))
            try:summary['conditions']['all_neutral_'+name]=evaluate(model,stores['val'],queries['val'])
            finally:hook.remove()
            summary['paired']['neutral_'+name]=grouped_comparison(summary['conditions']['all_intact']['rows'],
                summary['conditions']['all_neutral_'+name]['rows'])
            publish(root,report,summary)
        del model
        for mode in ('energy','both','envelope'):
            cfg=read(root/BASE/f'configs/{mode}.json')
            model=AcousticEncoder(**cfg['model']).cuda()
            state=torch.load(root/BASE/f'runs/{mode}/best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'])
            summary['conditions'][mode+'_trained']=evaluate(model,stores['val'],queries['val'])
            summary['paired'][mode+'_to_all']=grouped_comparison(summary['conditions'][mode+'_trained']['rows'],summary['conditions']['all_intact']['rows'])
            del model
            publish(root,report,summary)
        summary['status']='training near-capacity-matched E+C control'
        publish(root,report,summary)
        cfg=read(root/BASE/'configs/both.json');cfg=copy.deepcopy(cfg)
        cfg['model']['width']=40
        cfg['training'].update(workers=1,cpu_threads=1)
        cfg['wandb'].update(name='subset-ec-width40-capacity-control',group='three-input-mechanism-v1')
        immutable_json(out/'capacity_config.json',cfg)
        run_dir=out/'runs/ec_width40_seed42'
        if not (run_dir/'status.json').exists() or read(run_dir/'status.json')['status']!='complete':
            train(root,cfg,run_dir,run_dir/'last.pt' if (run_dir/'last.pt').exists() else None)
        state=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False)
        model=AcousticEncoder(**cfg['model']).cuda();model.load_state_dict(state['model'])
        summary['conditions']['both_width40_trained']=evaluate(model,stores['val'],queries['val'])
        summary['paired']['wide_both_to_all']=grouped_comparison(summary['conditions']['both_width40_trained']['rows'],summary['conditions']['all_intact']['rows'])
        summary.update(status='complete',capacity_best_epoch=state['epoch'],updated_unix=time.time())
        publish(root,report,summary)
    finally:
        for store in stores.values():store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.root,a.report)
