"""Four VAL-selected acoustic inputs on identical frozen dense TEST banks.

Reuses existing energy/coherence and common-envelope caches, not new RIRs.
Envelope is quantized to float16 exactly like its training input. No test fit.
"""
import argparse
from collections import Counter
import copy
import csv
import fcntl
import html
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .contracts import atomic_json, immutable_json, sha, recipe
from .data import read, splits
from .envelope_cache import extract as envelope_extract
from .evaluation import METHODS, diagnose, select, standardized, summarize
from .input_ablation import REL, MODES, LABELS, PILOT
from .pilot_evaluate import feature_arrays, heat_values, paired_intervals
from .pilot_report import build_report
from .train import AcousticEncoder, requires_envelope

BASE = PILOT/'corrected_yup_v2/evaluation'


def envelope_array(root, ready, handcrafted, scene, role, reference):
    entry = handcrafted['features'][scene][role]
    path = root/entry['path']
    if sha(path) != entry['sha256']:raise ValueError('Envelope receipt changed')
    receipt = read(path)
    if (receipt['paths'] != reference['paths'] or receipt['raw_receipt_sha256'] != ready['features'][scene][role]['sha256']
        or receipt['shapes']['common_arrival_L1'] != [6,252]):
        raise ValueError('Envelope and E/C inputs are not identical physical records')
    path = path.parent/'common_arrival_L1.npy'
    if sha(path) != receipt['files'][path.name]:raise ValueError('Envelope cache checksum changed')
    array = np.load(path,mmap_mode='r',allow_pickle=False).reshape(len(reference['paths']),6,252)
    for i in (0,len(array)//2,len(array)-1):
        raw_path = root/reference['paths'][i]
        if sha(raw_path) != reference['raw_sha256'][i]:raise ValueError('Pinned raw RIR changed')
        expected = envelope_extract(np.load(raw_path,allow_pickle=False)).astype(np.float16)
        np.testing.assert_array_equal(expected,array[i].astype(np.float16))
    return array


@torch.inference_mode()
def encode(model, e, c, envelope):
    values = [torch.as_tensor(np.ascontiguousarray(v,dtype=np.float32),device='cuda') for v in (e,c)]
    if envelope is not None:
        # Train caches are float16: do not silently supply higher precision here.
        values.append(torch.as_tensor(np.ascontiguousarray(envelope,dtype=np.float16).astype(np.float32),device='cuda'))
    return model(*values).cpu().numpy()


def embeddings(model, entry, fp, obs, envelopes, queries, batch):
    e,c,receipt = fp;oe,oc,oreceipt = obs;cells = entry['cells']
    if len(e) != cells*6:raise ValueError('Missing physical headings')
    candidate = np.empty((cells,36,128),np.float32)
    for yaw in range(36):
        base,roll = recipe(yaw*10)
        for first in range(0,cells,batch):
            ids = np.arange(first,min(first+batch,cells))*6+base//10
            env = np.roll(envelopes['FP'][ids],roll,axis=1) if envelopes else None
            candidate[first:first+len(ids),yaw] = encode(model,np.roll(e[ids],roll,axis=1),np.roll(c[ids],roll,axis=2),env)
    lut = {path:i for i,path in enumerate(oreceipt['paths'])};observed=[]
    for first in range(0,len(queries),batch):
        ev,cv,nv = [],[],[]
        for q in queries[first:first+batch]:
            i=lut[q['observed_rir_path']];roll=q['channel_roll']
            if roll != recipe(q['world_yaw_deg'])[1]:raise ValueError('Observed yaw/roll mismatch')
            ev.append(np.roll(oe[i],roll,axis=0));cv.append(np.roll(oc[i],roll,axis=1))
            if envelopes:nv.append(np.roll(envelopes['Observed'][i],roll,axis=0))
        observed.append(encode(model,np.stack(ev),np.stack(cv),np.stack(nv) if envelopes else None))
    return candidate,np.concatenate(observed)


def publish_comparison(report, result, modes=MODES, labels=LABELS, detail_links=None):
    rows=[]
    for mode in modes:
        if mode not in result['conditions']:continue
        for method,m in result['conditions'][mode]['micro'].items():
            if method=='vision_only' and mode!=modes[0]:continue
            values=[labels[mode] if method!='vision_only' else 'Common frozen vision',method,
                *(f'{100*m[k]:.2f}' for k in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg')),
                str(m['repair']),str(m['harm'])]
            rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HSSD acoustic input ablation · dense test</title><style>body{font:16px system-ui;margin:28px;color:#17314d;max-width:1450px}
td,th{border:1px solid #ccd6e5;padding:10px;text-align:right}td:first-child,td:nth-child(2){text-align:left}table{border-collapse:collapse;width:100%}
th{background:#edf3fc}.note{background:#fff1cf;padding:18px;line-height:1.7}a{color:#1260a3}.scroll{overflow:auto}</style>
<h1>HSSD · 입력 COUNT조건 dense test 비교</h1><p>상태: STATUS</p><div class="note">동일한 test 3개 scene / 1,472개 query.
0.2m XY grid, yaw 10°(36 bins). GT 후보 주입·query 필터링·test 기반 checkpoint/K/weight 선택 없음.
모델은 각 조건의 val loss로 이미 선택된 best를 사용합니다. Acoustic-only, 1:1 fusion(z-score), visual top-50 acoustic reranking을 비교합니다.
Train12/val3; 각 입력 모델의 파라미터 수가 다르고 single-seed입니다. 기존 captured-pose val 표와 다른 dense 평가입니다.
Test 전체 17 scene이 아니라 이전 subset의 3 scene입니다.</div><h2>조건별 RGB·likelihood·pose 상세</h2><ul>LINKS</ul>
<h2>동일 query 전체 · micro 평균 (%)</h2><div class="scroll"><table><tr><th>입력</th><th>방법</th><th>R@0.1m</th><th>R@0.5m</th><th>R@1m</th>
<th>R@1m·30°</th><th>Repair vs vision</th><th>Harm vs vision</th></tr>ROWS</table></div>
<p>각 상세 페이지에서 scene, query, likelihood 종류를 변경할 수 있습니다. GT·vision·acoustic·fusion·top-k는 색상 화살표로 표시됩니다.
Cosine scoring은 모든 조건에서 float64 CUDA로 동일하게 계산하고 NumPy와 교차 검증합니다.
Envelope는 기존 검증 캐시를 재사용하고 학습과 동일한 float16 입력으로 변환합니다.</p>
<p><a href="summary.json">전체 결과·입력 계약</a> · <a href="metrics.csv">CSV</a></p></html>'''
    links=''.join(f'<li><a href="{html.escape((detail_links or {}).get(mode,mode+"/"),quote=True)}">{html.escape(labels[mode])}</a></li>' for mode in modes if mode in result['conditions'])
    (report/'index.html').write_text(page.replace('COUNT',str(len(modes))).replace('STATUS',html.escape(result['status'])).replace('LINKS',links).replace('ROWS',''.join(rows)))
    atomic_json(report/'summary.json',result)
    with (report/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['input','policy','recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','repair','harm'])
        for mode,condition in result['conditions'].items():
            for method,m in condition['micro'].items():
                writer.writerow([mode,method,*[m[k] for k in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','repair','harm')]])


def run(root, vision, report, batch=256, ablation_rel=REL, modes=MODES, labels=LABELS):
    root,vision,report=map(Path,(root,vision,report));out=root/ablation_rel/'dense_test'
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    base_config=read(root/PILOT/'corrected_yup_v2/config.json')
    ready=read(root/BASE/'ready.json');index=read(root/BASE/'bank_index.json')
    if (ready['status']!='complete' or ready['bank_sha256']!=sha(root/BASE/'bank_index.json')
        or ready['config_sha256']!=sha(root/PILOT/'corrected_yup_v2/config.json')
        or {e['scene_id'] for e in index['scenes']}!=set(splits(root,base_config)['test'])):
        raise ValueError('Frozen test bank/split contract changed')
    trained=read(root/ablation_rel/'summary.json')
    if trained['status']!='complete' or trained['test_accessed']:raise ValueError('Val-only input ablation is not complete')
    hc_path=root/BASE/'handcrafted_comparison/ready.json';handcrafted=read(hc_path)
    old=vision/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/corrected-test'
    original=read(old/'records.json');lookup={(r['scene_id'],r['sample_index']):r for r in original}
    banks=read(old/'banks.json');reference=read(old/'summary.json')
    if len(original)!=1472 or reference['bank_sha256']!=sha(root/BASE/'bank_index.json'):
        raise ValueError('Original matched test report changed')
    contract=dict(bank_sha256=sha(root/BASE/'bank_index.json'),ready_sha256=sha(root/BASE/'ready.json'),
        handcrafted_ready_sha256=sha(hc_path),original_records_sha256=sha(old/'records.json'),
        ablation_summary_sha256=sha(root/ablation_rel/'summary.json'),models={m:sha(root/ablation_rel/'runs'/m/'best.pt') for m in modes},
        test_fitting=False,checkpoint_selection='frozen per-condition VAL loss best',
        scoring='float64 cosine over float32 normalized embeddings; no learned temperature',
        envelope_dtype='float16 quantization, matching training',new_RIRs=False,query_count=1472,
        grid_spacing_m=.2,yaw_step_deg=10,feature_definition=base_config['features'])
    visual_ready=read(root/'ReliabilityFusion/pilots/v1/test_scores_ready.json')
    visual_sources={}
    for e in visual_ready['entries']:
        folder=root/e['directory'];receipt=read(folder/'receipt.json')
        if sha(folder/'receipt.json')!=e['receipt_sha256']:raise ValueError('Frozen vision-score receipt changed')
        for name in ('scores.npy','xy.npy','truth.npy'):
            if sha(folder/name)!=receipt['files'][name]:raise ValueError('Frozen vision-score cache changed')
        visual_sources[e['scene_id']]=dict(directory=str(folder.relative_to(root)),receipt_sha256=e['receipt_sha256'])
    if set(visual_sources)!=set(splits(root,base_config)['test']):raise ValueError('Vision test scope changed')
    contract['visual_score_sources']=visual_sources
    immutable_json(out/'contract.json',contract)
    result=dict(status='running',conditions={},contract=contract,queries=1472,scenes=3)
    publish_comparison(report,result,modes,labels)
    for mode in modes:
        dest=report/mode;dest.mkdir(exist_ok=True);(dest/'heatmaps').mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            completed=read(dest/'complete.json')
            if completed['contract_sha256']!=sha(out/'contract.json'):raise ValueError('Stale completed input evaluation')
            result['conditions'][mode]=read(dest/'summary.json');publish_comparison(report,result,modes,labels);continue
        state=torch.load(root/ablation_rel/'runs'/mode/'best.pt',map_location='cpu',weights_only=False)
        config=state['config']
        if config['features']!=base_config['features'] or config['scene_subset_sha256']!=base_config['scene_subset_sha256']:
            raise ValueError('Model feature/split definition changed')
        model=AcousticEncoder(**config['model']).cuda().eval();model.load_state_dict(state['model'])
        records=[];per_scene={};checks=[]
        for entry in index['scenes']:
            scene=entry['scene_id'];xy=np.asarray(entry['candidate_xz'])
            fp=feature_arrays(root,ready,scene,'FP');obs=feature_arrays(root,ready,scene,'Observed')
            envelopes={role:envelope_array(root,ready,handcrafted,scene,role,values[2]) for role,values in [('FP',fp),('Observed',obs)]} if requires_envelope(mode) else None
            queries=read(root/entry['observation_schedule_path'])['queries']
            candidate,observed=embeddings(model,entry,fp,obs,envelopes,queries,batch)
            embedding_dir=out/'embeddings'/mode;embedding_dir.mkdir(parents=True,exist_ok=True)
            np.save(embedding_dir/(scene+'_candidate.npy'),candidate);np.save(embedding_dir/(scene+'_observed.npy'),observed)
            candidate_gpu=torch.tensor(candidate.reshape(-1,128),dtype=torch.float64,device='cuda')
            query_gpu=torch.tensor(observed,dtype=torch.float64,device='cuda')
            scene_records=[]
            for first in range(0,len(queries),8):
                audio=(query_gpu[first:first+8]@candidate_gpu.T).cpu().numpy().reshape(-1,len(xy),36)
                if first==0:
                    expected=np.einsum('cyd,d->cy',candidate.astype(np.float64),observed[0].astype(np.float64))
                    error=float(np.max(abs(expected-audio[0])))
                    if error>1e-10:raise ValueError('CUDA/NumPy cosine disagreement')
                    checks.append(dict(scene_id=scene,max_error=error))
                for j,a in enumerate(audio):
                    q=queries[first+j];source=lookup[(scene,q['sample_index'])];truth=source['gt_pose_xzyaw']
                    if not np.allclose(truth,[q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']],atol=1e-7):
                        raise ValueError('RGB/observed/GT query mismatch')
                    # Original frozen visual scores are already cached, with
                    # matching independent candidate XY/yaw order.
                    score_path=root/visual_sources[scene]['directory']/'scores.npy'
                    if j==0 and first==0:
                        vcache=np.load(score_path,mmap_mode='r');vxy=np.load(score_path.parent/'xy.npy')
                        vtruth=np.load(score_path.parent/'truth.npy')
                        np.testing.assert_allclose(vxy,xy,atol=1e-10)
                    np.testing.assert_allclose(vtruth[first+j],truth,atol=1e-7)
                    v=np.asarray(vcache[first+j,0])
                    row=diagnose(v,a,xy,truth,50)
                    np.testing.assert_allclose(row['methods']['vision_only']['pose_xzyaw'],source['methods']['vision_only']['pose_xzyaw'],atol=1e-7)
                    top=select(v,a,50)[1];topheat=np.full(len(xy),np.nan);topheat[top]=a.max(1)[top]
                    name=f'{scene}_{q["sample_index"]}.bin'
                    np.stack([heat_values(v.max(1)),heat_values(a.max(1)),heat_values((.5*standardized(v)+.5*standardized(a)).max(1)),heat_values(topheat)]).tofile(dest/'heatmaps'/name)
                    tags=[m+'_'+row['methods'][m]['vs_vision_1m'] for m in METHODS[1:]]
                    if not row['diagnostics']['topk_oracle_1m']:tags.append('topk_exclusion')
                    row.update({k:copy.deepcopy(source[k]) for k in ('scene_id','sample_index','gt_pose_xzyaw','image_path','existing_QA_flags',
                        'predicted_visual_rays','predicted_visual_scales','gt_nearest_fp_rays')})
                    row.update(tags=tags,heatmap='heatmaps/'+name,input_condition=mode)
                    records.append(row);scene_records.append(row)
            vcache._mmap.close();del candidate_gpu,query_gpu
            per_scene[scene]=summarize(scene_records)
            atomic_json(out/'status.json',dict(status='running',mode=mode,scene=scene,
                complete_conditions=len(result['conditions']),updated_unix=time.time()))
            print('evaluated',mode,scene,len(scene_records),flush=True)
        summary=dict(status='complete',micro=summarize(records),per_scene=per_scene,queries=len(records),scenes=3,
            acoustic_epoch=state['epoch'],acoustic_checkpoint_sha256=contract['models'][mode],tags=dict(Counter(t for r in records for t in r['tags'])),
            corrected_frame=True,input_condition=mode,numerical_checks=checks,
            paired_pose_bootstrap=paired_intervals(records),contract=contract,
            interpretations=[labels[mode]+' · 동일한 frozen dense test bank. Best는 test를 열기 전에 val loss로 선택했습니다.'],
            limitations=['Exploratory original3test scenes, not a new blind benchmark','Single seed; unequal parameter counts','0.2m grid limits R@0.1m'])
        for name in ('rgb','maps'):
            (dest/name).mkdir(exist_ok=True)
        for row in records:
            image=row['scene_id']+'_'+str(row['sample_index'])+'.png';target=dest/'rgb'/image
            if not target.exists():
                try:os.link(old/'rgb'/image,target)
                except OSError:shutil.copy2(old/'rgb'/image,target)
        atomic_json(dest/'summary.json',summary);atomic_json(dest/'records.json',records);atomic_json(dest/'banks.json',banks)
        with (dest/'metrics.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['method','recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg'])
            for m,vals in summary['micro'].items():writer.writerow([m,*[vals[k] for k in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg')]])
        build_report(root,dest,records,banks,summary)
        atomic_json(dest/'complete.json',dict(status='complete',contract_sha256=sha(out/'contract.json'),summary_sha256=sha(dest/'summary.json')))
        result['conditions'][mode]=summary;publish_comparison(report,result,modes,labels)
        del model;torch.cuda.empty_cache()
    result['status']='complete';publish_comparison(report,result,modes,labels)
    atomic_json(out/'summary.json',result);atomic_json(out/'status.json',dict(status='complete',queries=1472,
        conditions=len(modes),report=str(report),test_fitting=False,updated_unix=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','vision','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=256);args=p.parse_args()
    run(args.root,args.vision,args.report,args.batch_size)
