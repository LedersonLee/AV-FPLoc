"""Best/Last global acoustic localization on all held-out VAL scenes only.

Stream each FP RIR once into three checked features, encode both frozen models,
and match against the same GT-independent 0.2m / 10-degree bank. No test arrays.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import html
import io
from pathlib import Path
import time
import traceback

import numpy as np

from .contracts import PROTOCOL,atomic_json,immutable_json,sha,recipe
from .data import read,load_manifest,check_cache
from .dense_banks import REL as BANK
from .envelope_cache import extract as envelope_extract,DEFINITION
from .features import extract,feature_id
from .full_scene_workflow import REL as FULL

REL=FULL+'/dense_val_best_last_v1'
REVISION='ReliabilityFusion/pilots/ecn14_v1'
MODES=('best','last')


def frozen_contract(root):
    root=Path(root);run=root/FULL/'runs/energy_coherence_envelope_seed42'
    config=read(root/FULL/'config.json');index=read(root/BANK/'bank_index.json')
    membership=read(root/'manifests/scene_splits_v1.json');ready=read(root/BANK/'ready.json')
    if ready['status']!='complete' or ready['bank_sha256']!=sha(root/BANK/'bank_index.json'):raise ValueError('Dense bank incomplete')
    scenes=[e for e in index['scenes'] if e['split']=='val']
    if {e['scene_id'] for e in scenes}!=set(membership['val']) or len(scenes)!=19:raise ValueError('Need all19 val scenes')
    if config['split_sha256']!=sha(root/'manifests/scene_splits_v1.json'):raise ValueError('Parent split changed')
    if read(run/'status.json')['status']!='complete':raise ValueError('Training must be complete')
    sources={}
    for e in scenes:
        path=root/BANK/'verified/val'/(e['scene_id']+'.json');receipt=read(path)
        if (receipt['status']!='complete' or receipt['split']!='val' or receipt['fp_arrays']!=e['fp_arrays']
                or receipt['schedule_sha256']!=e['schedule_sha256'] or sha(root/e['schedule_path'])!=e['schedule_sha256']):
            raise ValueError('VAL raw receipt/schedule changed')
        sources[e['scene_id']]=dict(receipt_path=str(path.relative_to(root)),receipt_sha256=sha(path),
            schedule_sha256=e['schedule_sha256'])
    contract=dict(version='full_dense_val_best_last_v1',config_sha256=sha(root/FULL/'config.json'),
        split_sha256=config['split_sha256'],dense_index_sha256=sha(root/BANK/'bank_index.json'),
        checkpoints={n:sha(run/(n+'.pt')) for n in MODES},sources=sources,scenes=19,queries=9965,
        cells=sum(e['cells'] for e in scenes),fp_arrays=sum(e['fp_arrays'] for e in scenes),
        grid_m=.2,yaw_step_deg=10,feature_precision='float16 quantization matching training',
        scoring='float64 dot product of float32 unit embeddings; deterministic bank-order ties',
        model_selection='No retraining or test fitting; fixed completed best and last',test_accessed=False)
    return config,scenes,contract


def metrics(distance,yaw):
    distance,yaw=np.asarray(distance),np.asarray(yaw)
    out={f'recall_{t:g}m':float((distance<=t).mean()) for t in (.1,.5,1)}
    out.update(recall_1m_30deg=float(((distance<=1)&(yaw<=30)).mean()),median_xy_m=float(np.median(distance)),
        median_yaw_deg=float(np.median(yaw)),queries=len(distance))
    out.update({f'yaw_within_{t}deg':float((yaw<=t).mean()) for t in (10,30,50)})
    return out


def assemble(shards,cells):
    """Map raw-index[base] + physical mic roll to the world-yaw candidate grid."""
    candidate=np.empty((cells,36,128),np.float32);covered=np.zeros((cells,36),bool)
    for first,values in shards:
        for yaw in range(36):
            base,roll=recipe(yaw*10)
            raw=np.arange(first,first+len(values));selected=raw%6==base//10
            candidate[raw[selected]//6,yaw]=values[selected,roll%6]
            covered[raw[selected]//6,yaw]=True
    if not covered.all() or not np.isfinite(candidate).all():raise ValueError('Incomplete candidate yaw support')
    return candidate


def encode_both(models,values):
    import torch
    e,c,n=[torch.tensor(np.ascontiguousarray(v,dtype=np.float32),device='cuda') for v in values]
    output={m:np.empty((len(e),6,128),np.float32) for m in MODES}
    with torch.inference_mode():
        for roll in range(6):
            inputs=(torch.roll(e,roll,1),torch.roll(c,roll,2),torch.roll(n,roll,1))
            for name,model in models.items():output[name][:,roll]=model(*inputs).cpu().numpy()
    return output


def reuse_features(root,scene,records):
    """Alias the completed fusion VAL feature bytes; never use train/test caches."""
    path=root/REVISION/'envelope_ready.json'
    if not path.exists():return None
    ready=read(path);matches=[e for e in ready['entries'] if e['split']=='val' and e['scene_id']==scene and e['role']=='FP']
    if not matches:return None
    entry=matches[0];old=root/'ReliabilityFusion/pilots/v1/ready.json';ref=read(old)['features'][scene]['FP']
    rp=root/ref['path'];r=read(rp)
    if sha(rp)!=ref['sha256']:raise ValueError('Reusable E/C receipt changed')
    if r['paths']!=[p['path'] for p in records] or r['raw_sha256']!=[p['sha256'] for p in records]:
        raise ValueError('Reusable feature records differ from dense VAL bank')
    values=[]
    for name in ('energy.npy','coherence.npy'):
        p=rp.parent/name
        if sha(p)!=r['files'][name]:raise ValueError('Reusable E/C data changed')
        values.append(np.load(p,mmap_mode='r'))
    envelopes=[]
    for chunk in entry['chunks']:
        p=root/chunk['path'];receipt=read(p.parent/'receipt.json')
        if sha(p.parent/'receipt.json')!=chunk['receipt_sha256'] or sha(p)!=receipt['sha256']:raise ValueError('Reusable envelope changed')
        envelopes.append(np.load(p,allow_pickle=False))
    values.append(np.concatenate(envelopes))
    return values


def fp_embeddings(root,out,entry,config,models,contract,status,workers):
    scene=entry['scene_id'];source=contract['sources'][scene];rp=root/source['receipt_path']
    if sha(rp)!=source['receipt_sha256']:raise ValueError('Dense VAL raw receipt changed')
    records=read(rp)['records'];schedule=read(root/entry['schedule_path'])
    expected=[f'RIR/FP/val/{scene}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_{b:03d}.npy'
        for p in schedule['positions'] for b in range(0,60,10)]
    if expected!=[r['path'] for r in records]:raise ValueError('Physical base order changed')
    cache=root/'AcousticContrastive/features'/feature_id(config['features'])/'dense_val'/contract['dense_index_sha256'][:12]/scene
    pairs={n:[] for n in MODES};reused=None
    # Only initialize the alias when actually needed; complete embedding shards
    # can be reused without rereading a feature payload.
    def work(index):
        record=records[index];path=root/record['path'];meta=read(path.with_suffix('.json'))
        if (meta['split'],meta['scene_id'],meta['role'])!=('val',scene,'FP') or meta['acoustics']!=PROTOCOL:
            raise ValueError('FP identity/protocol mismatch')
        p=schedule['positions'][index//6];base=index%6*10;frame=meta.get('loaded_frame') or {}
        if (not frame.get('verified') or frame.get('up')!=[0,1,0] or frame.get('front')!=[0,0,-1]
                or meta['geometry_sha256']!=schedule['geometry_sha256'] or meta['rir_sha256']!=record['sha256']):
            raise ValueError('Uncorrected or stale FP RIR')
        np.testing.assert_allclose(meta['centre_xyz_m'],p['centre_xyz_m'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(frame['loaded_bounds'],schedule['height_contract']['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
        if int(path.stem.rsplit('_',1)[1])!=base:raise ValueError('Physical base heading mismatch')
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError('FP raw bytes changed')
        raw=np.load(io.BytesIO(payload),allow_pickle=False);e,c=extract(raw,config['features']);n=envelope_extract(raw)
        return e.astype(np.float16),c.astype(np.float16),n.astype(np.float16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,len(records),256):
            end=min(first+256,len(records));folder=out/'embeddings'/scene/f'{first//256:05d}';folder.mkdir(parents=True,exist_ok=True)
            receipt=folder/'receipt.json'
            shard_contract=dict(contract_sha256=sha(out/'contract.json'),first=first,count=end-first,
                source_receipt_sha256=source['receipt_sha256'])
            if receipt.exists():
                row=read(receipt)
                if row['contract']!=shard_contract:raise ValueError('Stale embedding shard')
                for name,digest in row['files'].items():
                    if sha(folder/name)!=digest:raise ValueError('Embedding bytes changed')
            else:
                feature_folder=cache/f'{first//256:05d}';fr=feature_folder/'receipt.json'
                fc=dict(records=records[first:end],features=config['features'],envelope=DEFINITION,split='val',scene_id=scene)
                if fr.exists():
                    rr=read(fr)
                    if rr['contract']!=fc:raise ValueError('Dense feature contract changed')
                    arrays=[]
                    for name,digest in rr['files'].items():
                        if sha(feature_folder/name)!=digest:raise ValueError('Dense feature cache changed')
                    arrays=[np.load(feature_folder/(name+'.npy')) for name in ('energy','coherence','envelope')]
                else:
                    if first==0:reused=reuse_features(root,scene,records)
                    if reused is not None:arrays=[np.asarray(v[first:end],dtype=np.float16) for v in reused]
                    else:
                        chunk=list(pool.map(work,range(first,end)))
                        arrays=[np.stack([v[k] for v in chunk]) for k in range(3)]
                    feature_folder.mkdir(parents=True,exist_ok=True)
                    for name,values in zip(('energy','coherence','envelope'),arrays):
                        np.save(feature_folder/(name+'.tmp.npy'),values,allow_pickle=False)
                        (feature_folder/(name+'.tmp.npy')).replace(feature_folder/(name+'.npy'))
                    immutable_json(fr,dict(contract=fc,files={name+'.npy':sha(feature_folder/(name+'.npy')) for name in ('energy','coherence','envelope')}))
                encoded=encode_both(models,arrays)
                for name,values in encoded.items():
                    np.save(folder/(name+'.tmp.npy'),values);(folder/(name+'.tmp.npy')).replace(folder/(name+'.npy'))
                immutable_json(receipt,dict(contract=shard_contract,files={n+'.npy':sha(folder/(n+'.npy')) for n in MODES}))
            for name in MODES:pairs[name].append((first,np.load(folder/(name+'.npy'),allow_pickle=False)))
            status('candidate_embeddings',scene_id=scene,physical_records=end,scene_physical_total=len(records),
                scene_uses_existing_features=reused is not None)
    return {name:assemble(values,entry['cells']) for name,values in pairs.items()},schedule


def summarize(rows):
    result={name:metrics([r[name]['xy_error_m'] for r in rows],[r[name]['yaw_error_deg'] for r in rows]) for name in MODES}
    before=np.array([r['best']['xy_error_m']<=1 for r in rows]);after=np.array([r['last']['xy_error_m']<=1 for r in rows])
    return dict(models=result,last_vs_best=dict(repair=int((~before&after).sum()),harm=int((before&~after).sum())))


def publish(out,report,entries,contract,status):
    rows=[];per_scene={}
    for entry in entries:
        path=out/'scenes'/(entry['scene_id']+'.json')
        if path.exists():
            document=read(path)
            if document['contract_sha256']!=sha(out/'contract.json'):raise ValueError('Scene result contract changed')
            rows.extend(document['records']);per_scene[entry['scene_id']]=summarize(document['records'])
    summary=dict(status=status,completed_scenes=len(per_scene),total_scenes=19,completed_queries=len(rows),total_queries=9965,
        contract=contract,per_scene=per_scene,test_accessed=False)
    if rows:
        summary.update(summarize(rows));summary['scene_macro']={n:{key:float(np.mean([v['models'][n][key] for v in per_scene.values()]))
            for key in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg')} for n in MODES}
    atomic_json(out/'summary.json',summary);atomic_json(report/'summary.json',summary)
    fields=['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','yaw_within_10deg','yaw_within_30deg','yaw_within_50deg','median_xy_m','median_yaw_deg']
    if rows:
        with (report/'metrics.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['scope','checkpoint',*fields])
            for scope,values in [('micro',summary),*per_scene.items()]:
                for name in MODES:writer.writerow([scope,name,*[values['models'][name][k] for k in fields]])
        atomic_json(report/'records.json',rows)
    body=[]
    for scope,values in ([('micro',summary),*per_scene.items()] if rows else []):
        for name in MODES:
            m=values['models'][name]
            body.append('<tr><td>'+html.escape(scope)+'</td><td>'+name+'</td>'+''.join('<td>'+
                (f'{m[k]*100:.2f}%' if k.startswith(('recall','yaw_within')) else f'{m[k]:.2f}')+'</td>' for k in fields)+'</tr>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Full acoustic · Best / Last dense VAL</title><style>body{font:16px system-ui;margin:28px;color:#19314a}p{line-height:1.65}.scroll{overflow:auto}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #d4dce5;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap}</style>
<h1>전체 scene acoustic · Best(3epoch) / Last(11epoch) dense VAL</h1><p>STATUS · SCENES/19 scene · QUERIES/9,965 queries.
진행 중 수치는 완료된 scene만의 부분 집계입니다. 전체 완료 전 성능을 확정하지 마세요.</p>
<p>동일한 독립 0.2m×0.2m / yaw10° bank, FP 3D 압출·천장 포함. 전체 train116 학습 모델 두 개를 고정합니다.
Val19만 평가하며 test 배열·성능은 읽지 않습니다. 후보에 GT pose를 주입하지 않습니다.
R.1/.5/1은 XY만, R1+30°는 XY·yaw 공동 성공률입니다. 8후보 또는 촬영-pose bank 성능과 다릅니다.</p>
<p><a href="summary.json">전체 결과·프로토콜 JSON</a> · <a href="metrics.csv">CSV</a> · <a href="records.json">Query별 pose·오차</a> · <a href="status.json">실행 상태</a></p>
<div class="scroll"><table><tr><th>집계</th><th>Checkpoint</th><th>R.1</th><th>R.5</th><th>R1</th><th>R1+30°</th><th>Yaw≤10°</th><th>Yaw≤30°</th><th>Yaw≤50°</th><th>XY 중앙값(m)</th><th>Yaw 중앙값(°)</th></tr>ROWS</table></div>
<p>Micro는 모든 query를 동일 가중치로 집계하며 scene macro는 JSON에 별도로 저장합니다.
Last의 repair/harm은 Best 대비 위치 1m 성공 여부 변화이며 vision 대비 수치가 아닙니다.</p>
<script>if('STATUS'!=='complete')setTimeout(()=>location.reload(),30000);</script></html>'''
    (report/'index.html').write_text(page.replace('STATUS',status).replace('SCENES',str(len(per_scene))).replace('QUERIES',str(len(rows))).replace('ROWS',''.join(body)))


def run(root,report,workers=48):
    import torch
    from .train import AcousticEncoder,FeatureStore,encode_refs
    root,report=Path(root),Path(report);out=root/REL;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config,entries,contract=frozen_contract(root);immutable_json(out/'contract.json',contract)
    def status(stage,**kwargs):
        value=dict(status='running',stage=stage,updated_unix=time.time(),test_accessed=False,**kwargs)
        atomic_json(out/'status.json',value);atomic_json(report/'status.json',value)
    publish(out,report,entries,contract,'running');status('verify_observed_val_cache')
    store=None
    try:
        queries,digest=load_manifest(root,'val',config)
        if len(queries)!=9965:raise ValueError('Val query count changed')
        locations,_=check_cache(root,config,'val',digest,full=True)
        envelopes={path:(str(Path(folder)/'envelope.npy'),index) for path,(folder,index) in locations.items()}
        store=FeatureStore(locations,max_open=32,envelope_locations=envelopes)
        torch.set_num_threads(4);models={};run_dir=root/FULL/'runs/energy_coherence_envelope_seed42'
        for name in MODES:
            path=run_dir/(name+'.pt')
            if sha(path)!=contract['checkpoints'][name]:raise ValueError('Checkpoint changed')
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['config']!=config:raise ValueError('Model config mismatch')
            model=AcousticEncoder(**config['model']).cuda().eval();model.load_state_dict(state['model']);models[name]=model
            if state['epoch']!=(3 if name=='best' else 11):raise ValueError('Expected fixed epoch3/11 comparison')
        # Reuse three completed fusion val feature streams first, then small-to-large
        # remaining banks; order has no effect on the final fixed all19 comparison.
        reusable=set()
        if (root/REVISION/'envelope_ready.json').exists():
            reusable={e['scene_id'] for e in read(root/REVISION/'envelope_ready.json')['entries'] if e['split']=='val'}
        ordered=sorted(entries,key=lambda e:(e['scene_id'] not in reusable,e['cells'],e['scene_id']))
        for entry in ordered:
            scene=entry['scene_id'];dest=out/'scenes'/(scene+'.json')
            if dest.exists():publish(out,report,entries,contract,'running');continue
            candidate,schedule=fp_embeddings(root,out,entry,config,models,contract,status,workers)
            xy=np.array([[p['centre_xyz_m'][0],p['centre_xyz_m'][2]] for p in schedule['positions']])
            group=[q for q in queries if q['scene_id']==scene];refs=[q['observed'] for q in group]
            observed={n:encode_refs(m,store,refs,torch.device('cuda'),128).numpy() for n,m in models.items()}
            gt=np.array([[r['xyz'][0],r['xyz'][2],r['yaw']] for r in refs]);rows=[]
            for name in MODES:
                cg=torch.tensor(candidate[name].reshape(-1,128),dtype=torch.float64,device='cuda')
                winners=[];check_error=None
                for first in range(0,len(group),8):
                    scores=(torch.tensor(observed[name][first:first+8],dtype=torch.float64,device='cuda')@cg.T).cpu().numpy()
                    if first==0:
                        reference=candidate[name].astype(np.float64).reshape(-1,128)@observed[name][0].astype(np.float64)
                        check_error=float(np.max(abs(reference-scores[0])))
                        if check_error>1e-10:raise ValueError('NumPy/CUDA cosine mismatch')
                    winners.extend(scores.argmax(1).tolist())
                ids=np.asarray(winners);poses=np.column_stack([xy[ids//36],ids%36*10])
                distance=np.linalg.norm(poses[:,:2]-gt[:,:2],axis=1);yaw=abs((poses[:,2]-gt[:,2]+180)%360-180)
                if not rows:rows=[dict(scene_id=scene,sample_index=q['sample_index'],gt_pose_xzyaw=gt[i].tolist()) for i,q in enumerate(group)]
                for i,row in enumerate(rows):row[name]=dict(pose_xzyaw=poses[i].tolist(),xy_error_m=float(distance[i]),yaw_error_deg=float(yaw[i]))
                del cg
            oracle=np.linalg.norm(xy[None,:,:]-gt[:,None,:2],axis=2).min(1)
            atomic_json(dest,dict(contract_sha256=sha(out/'contract.json'),records=rows,queries=len(group),
                candidate_oracle_1m=float((oracle<=1).mean()),max_quantization_m=float(oracle.max()),test_accessed=False))
            publish(out,report,entries,contract,'running');status('scene_complete',scene_id=scene,queries=len(group))
            print('dense_val_scene',scene,summarize(rows),flush=True)
        publish(out,report,entries,contract,'complete')
        summary=read(out/'summary.json')
        if summary['completed_queries']!=9965 or summary['completed_scenes']!=19:raise ValueError('Incomplete final VAL coverage')
        atomic_json(out/'status.json',dict(status='complete',test_accessed=False,updated_unix=time.time()))
        atomic_json(report/'status.json',read(out/'status.json'))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',test_accessed=False,error=traceback.format_exc(),updated_unix=time.time()))
        atomic_json(report/'status.json',read(out/'status.json'));raise
    finally:
        if store:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--workers',type=int,default=48)
    a=p.parse_args();run(a.root,a.report,a.workers)
