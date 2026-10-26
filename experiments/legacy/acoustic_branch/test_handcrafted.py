"""Frozen TEST comparison: learned audio vs three predeclared VAL methods.

No RIR rendering, retraining, test-fit parameters, or candidate filtering.
CPU extraction preserves original float32 features; CUDA distances use float64
and are checked against SciPy cdist. Node2 never runs GPU code.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from numpy.lib.format import open_memmap

from .contracts import atomic_json,immutable_json,sha,recipe,PROTOCOL
from .data import read,safe_path
from .features import extract
from .planar_probe import arrival_features

BASE='AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2'
REL=BASE+'/evaluation/handcrafted_comparison'
METHODS=('STFT_L1','coherence_L2','common_arrival_L1')
LABELS=dict(learned='Learned acoustic',STFT_L1='STFT-L1',coherence_L2='Complex coherence',
            common_arrival_L1='Common-removed envelope')


def reference(source):
    spec=importlib.util.spec_from_file_location('frozen_grid_score',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    settings=module.GridScoreConfig(sample_rate_hz=48000,direct_guard_samples=96,usable_samples=6144,
        nfft=256,hop=64,bands=((0,500),(500,1500),(1500,4000)),normalise='sum',distance='symmetric',
        shape_feature=True,feature='stft_band')
    return module,settings


def features(rir,config,module,settings):
    stft=module.featurise(module.band_energy(rir,settings),settings).astype(np.float32)
    try:coherence=extract(rir,config)[1]
    except ValueError as error:
        if 'No reflection energy' not in str(error):raise
        coherence=np.zeros((2,6,2,19),np.float32)
    residual=rir.astype(np.float64)-rir.mean(axis=0,dtype=np.float64)
    envelope=arrival_features(residual)[0]
    return dict(STFT_L1=stft,coherence_L2=coherence,common_arrival_L1=envelope)


def roll(array,method,amount):
    return np.roll(array.reshape(6,3,-1),amount,0).reshape(array.shape) if method=='STFT_L1' else np.roll(array,amount,1 if method=='coherence_L2' else 0)


def initialize_feature_worker(root,config,source,role,scene):
    global FEATURE_WORKER
    module,settings=reference(Path(source))
    FEATURE_WORKER=(Path(root),config,module,settings,role,scene)


def feature_worker(item):
    root,config,module,settings,role,scene=FEATURE_WORKER
    path,digest=item;relative=Path(path)
    if (relative.is_absolute() or '..' in relative.parts
        or tuple(relative.parts[:4])!=('RIR',role,'test',scene)):
        raise ValueError('Raw path outside frozen test role/scene')
    p=root/relative;meta=read(p.with_suffix('.json'))
    if (meta['role']!=role or meta['split']!='test' or meta['scene_id']!=scene
        or meta['acoustics']!=PROTOCOL or meta['rir_sha256']!=digest or sha(p)!=digest):
        raise ValueError('Raw identity/SHA changed')
    if role=='FP' and not (meta.get('loaded_frame') or {}).get('verified'):
        raise ValueError('Unverified FP frame')
    result=features(np.load(p,allow_pickle=False),config,module,settings)
    if not all(np.isfinite(array).all() for array in result.values()):raise ValueError('Nonfinite handcrafted feature')
    return result


def prepare(root,source):
    out=root/REL
    prior=root/BASE/'evaluation'
    ready=read(prior/'ready.json');index=read(prior/'bank_index.json')
    if ready['status']!='complete' or ready['frame_version']!='corrected_yup_v2' or ready['bank_sha256']!=sha(prior/'bank_index.json'):
        raise ValueError('Corrected test bank is not ready')
    val=read(root/'AcousticContrastive/diagnostics/planar_common_v2/results.json')
    if val['source_sha256']!=sha(source):raise ValueError('STFT reference differs from earlier VAL comparison')
    config=root/BASE/'config.json'
    learned=root/BASE/'evaluation/report/summary.json'
    if read(learned)['status']!='complete':raise ValueError('Learned test reference incomplete')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,out/'grid_score.py')
    contract=dict(version='test_handcrafted_comparison_v1',bank_sha256=sha(prior/'bank_index.json'),
        ready_sha256=sha(prior/'ready.json'),config_sha256=sha(config),learned_summary_sha256=sha(learned),
        learned_records_sha256=sha(prior/'report/records.json'),
        scoring_source_sha256=sha(source),methods=list(METHODS),
        feature_code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('test_handcrafted.py','planar_probe.py','features.py','evaluation.py')},
        scenes=[e['scene_id'] for e in index['scenes']],queries=sum(e['queries'] for e in index['scenes']),
        geometry='same corrected canonical 2DFP extrusion RIRs',grid_spacing_m=.2,yaw_step_deg=10,
        no_test_fitting=True,new_RIRs=False,retraining=False,feature_dtype='float32, not quantized learned feature cache',
        scoring_dtype='float64 CUDA; checked against float64 SciPy cdist',
        STFT_L1='same legacy 0-500/500-1500/1500-4000Hz, nfft256 hop64, peak+96 guard, 6144 samples, sum/shape feature',
        coherence_L2='same complex coherency Re/Im, 500-4000Hz, fixed direct7, 2-50/2-128ms, squared L2',
        common_arrival_L1='same per-RIR six-mic mean subtraction; fixed clock/direct7+2ms; 500-1500Hz Hilbert/Gaussian envelope; .5ms bins; per-mic peak norm; L1',
        policies=['acoustic_only','equal_weight_fusion','visual_topk_acoustic_rerank'],primary_k=50,
        tie_policy='same fixed independent dense-bank order as learned evaluation; no GT tie preference')
    immutable_json(out/'contract.json',contract)
    print(dict(status='prepared',queries=contract['queries'],methods=list(METHODS)),flush=True)


def extract_cache(a):
    out=a.root/REL;config=read(a.root/BASE/'config.json');contract=read(out/'contract.json')
    if sha(out/'grid_score.py')!=contract['scoring_source_sha256']:raise ValueError('Reference changed')
    prior=a.root/BASE/'evaluation';index=read(prior/'bank_index.json');ready=read(prior/'ready.json')
    if (sha(prior/'bank_index.json')!=contract['bank_sha256'] or sha(prior/'ready.json')!=contract['ready_sha256']
        or sha(a.root/BASE/'config.json')!=contract['config_sha256']):raise ValueError('Frozen comparison inputs changed')
    lock=(out/'extract.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts={}
    for entry in index['scenes']:
        scene=entry['scene_id'];receipts[scene]={}
        for role in ('FP','Observed'):
            raw_receipt=safe_path(a.root,ready['features'][scene][role]['path'])
            if sha(raw_receipt)!=ready['features'][scene][role]['sha256']:raise ValueError('Test raw receipt changed')
            raw=read(raw_receipt);paths=raw['paths']
            dest=out/'features'/scene/role;dest.mkdir(parents=True,exist_ok=True)
            receipt_path=dest/'receipt.json'
            if receipt_path.exists():
                receipt=read(receipt_path)
                if receipt['contract_sha256']!=sha(out/'contract.json') or receipt['raw_receipt_sha256']!=sha(raw_receipt):raise ValueError('Stale handcrafted cache')
                for name,digest in receipt['files'].items():
                    if sha(dest/name)!=digest:raise ValueError('Corrupt handcrafted cache')
                receipts[scene][role]=dict(path=str(receipt_path.relative_to(a.root)),sha256=sha(receipt_path));continue
            arrays={};shapes={}
            with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize_feature_worker,
                initargs=(str(a.root),config['features'],str(out/'grid_score.py'),role,scene)) as pool:
                for i,value in enumerate(pool.map(feature_worker,zip(paths,raw['raw_sha256']),chunksize=8)):
                    if not arrays:
                        for method,array in value.items():
                            shapes[method]=list(array.shape)
                            arrays[method]=open_memmap(dest/(method+'.tmp.npy'),mode='w+',dtype='float32',shape=(len(paths),array.size))
                    for method,array in value.items():arrays[method][i]=array.ravel()
                    if i%256==0:
                        atomic_json(out/'status.json',dict(status='running',stage='extract_features',scene=scene,role=role,complete=i,total=len(paths),updated_unix=time.time()))
            files={}
            for method,array in arrays.items():
                array.flush();array._mmap.close();(dest/(method+'.tmp.npy')).replace(dest/(method+'.npy'));files[method+'.npy']=sha(dest/(method+'.npy'))
            immutable_json(receipt_path,dict(paths=paths,shapes=shapes,files=files,contract_sha256=sha(out/'contract.json'),raw_receipt_sha256=sha(raw_receipt)))
            receipts[scene][role]=dict(path=str(receipt_path.relative_to(a.root)),sha256=sha(receipt_path))
            print(scene,role,'feature cache complete',len(paths),flush=True)
    atomic_json(out/'ready.json',dict(status='complete',contract_sha256=sha(out/'contract.json'),features=receipts))
    atomic_json(out/'status.json',dict(status='complete',stage='handcrafted_features_ready',updated_unix=time.time()))


def load_features(root,ready,scene,role):
    item=ready['features'][scene][role];path=safe_path(root,item['path'])
    if sha(path)!=item['sha256']:raise ValueError('Handcrafted receipt changed')
    receipt=read(path)
    for name,digest in receipt['files'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Handcrafted feature SHA changed')
    return {m:np.load(path.parent/(m+'.npy'),mmap_mode='r') for m in METHODS},receipt


def dense_features(array,shape,method,cells):
    # Keep cell-major/yaw-minor order identical to learned embeddings.
    bank=np.empty((cells,36,array.shape[-1]),np.float64)
    for yaw in range(36):
        base,amount=recipe(yaw*10)
        value=np.array(array[np.arange(cells)*6+base//10],np.float64).reshape(cells,*shape)
        axis=2 if method=='coherence_L2' else 1
        if method=='STFT_L1':value=value.reshape(cells,6,3,-1)
        bank[:,yaw]=np.roll(value,amount,axis=axis).reshape(cells,-1)
    return bank.reshape(cells*36,-1)


def evaluate(a):
    import torch
    from scipy.spatial.distance import cdist
    from .evaluation import diagnose,summarize,select,standardized
    from .pilot_evaluate import angular_rays,bank_windows,heat_values,paired_intervals
    out=a.root/REL;prior=a.root/BASE/'evaluation';a.report.mkdir(parents=True,exist_ok=True)
    lock=(out/'evaluate.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=read(out/'status.json') if (out/'status.json').exists() else {}
        if status.get('status')=='failed':raise RuntimeError('Handcrafted extraction failed')
        if (out/'ready.json').exists():break
        atomic_json(a.report/'status.json',dict(status='waiting_features',progress=status,updated_unix=time.time()))
        time.sleep(30)
    ready=read(out/'ready.json');contract=read(out/'contract.json')
    if ready['contract_sha256']!=sha(out/'contract.json'):raise ValueError('Comparison contract changed')
    learned=read(prior/'report/summary.json');learned_records=read(prior/'report/records.json')
    if (sha(prior/'report/summary.json')!=contract['learned_summary_sha256']
        or sha(prior/'report/records.json')!=contract['learned_records_sha256']
        or sha(prior/'bank_index.json')!=contract['bank_sha256']):raise ValueError('Comparison reference changed')
    lut={(r['scene_id'],r['sample_index']):r for r in learned_records}
    index=read(prior/'bank_index.json');torch.set_num_threads(4)
    records={m:[] for m in METHODS};banks=read(prior/'report/banks.json')
    numeric_checks=[]
    for entry in index['scenes']:
        scene=entry['scene_id'];xy=np.array(entry['candidate_xz']);cells=len(xy)
        fp,fpr=load_features(a.root,ready,scene,'FP');obs,obsr=load_features(a.root,ready,scene,'Observed')
        queries=read(a.root/entry['observation_schedule_path'])['queries'];obs_lut={p:i for i,p in enumerate(obsr['paths'])}
        depth_path=a.vision_eval/'depth'/f'{scene}.npz';rows_path=a.vision_eval/'rows'/f'{scene}.json'
        payload_path=a.vision_hssd/'UnLoc/structural_targets'/scene/'desdf.npy'
        frozen=learned['frozen_inputs'][scene]
        for path,key in ((depth_path,'depth_sha256'),(rows_path,'rows_sha256'),(payload_path,'desdf_sha256')):
            if sha(path)!=frozen[key]:raise ValueError('Frozen visual input changed')
        rows=read(rows_path);row_lut={r['sample_index']:i for i,r in enumerate(rows)}
        with np.load(depth_path) as blob:depth=blob['location'].copy();scale=blob['scale'].copy()
        windows=bank_windows(np.load(payload_path,allow_pickle=True).item(),xy)
        visual=[]
        for q in queries:
            j=row_lut[q['sample_index']];rays,scales=angular_rays(depth[j],scale[j]);visual.append(-(abs(windows-rays)/scales).sum(-1)/40)
        for method in METHODS:
            target=a.report/method;target.mkdir(exist_ok=True);(target/'heatmaps').mkdir(exist_ok=True)
            candidate=dense_features(fp[method],fpr['shapes'][method],method,cells)
            observed=np.stack([roll(obs[method][obs_lut[q['observed_rir_path']]].reshape(obsr['shapes'][method]),method,q['channel_roll']).ravel() for q in queries]).astype(np.float64)
            bank_tensor=torch.from_numpy(candidate).to('cuda');p=2 if method=='coherence_L2' else 1
            for first in range(0,len(queries),a.batch_size):
                values=torch.from_numpy(observed[first:first+a.batch_size]).to('cuda')
                with torch.inference_mode():
                    d=torch.cdist(values,bank_tensor,p=p,compute_mode='donot_use_mm_for_euclid_dist')
                    if p==2:d=d.square()
                distances=d.cpu().numpy()
                if first==0:
                    expected=cdist(observed[:min(3,len(values))],candidate[:11],metric='sqeuclidean' if p==2 else 'cityblock')
                    np.testing.assert_allclose(distances[:len(expected),:11],expected,atol=1e-8,rtol=1e-10)
                    numeric_checks.append(dict(scene=scene,method=method,max_abs_error=float(abs(distances[:len(expected),:11]-expected).max())))
                for offset,row in enumerate(distances):
                    i=first+offset;q=queries[i];old=lut[scene,q['sample_index']];v=visual[i];audio=-row.reshape(cells,36)
                    truth=old['gt_pose_xzyaw'];result=diagnose(v,audio,xy,truth,50)
                    if result['methods']['vision_only']!=old['methods']['vision_only']:raise ValueError('Vision baseline not identical')
                    tags=[]
                    if not result['diagnostics']['topk_oracle_1m']:tags.append('topk_exclusion')
                    elif result['methods']['visual_topk_acoustic_rerank']['xy_error_m']>1:tags.append('acoustic_selection_failure')
                    for policy,r in result['methods'].items():
                        if r['vs_vision_1m'] in ('repair','harm'):tags.append(policy+'_'+r['vs_vision_1m'])
                        if r['xy_error_m']<=1 and r['yaw_error_deg']>30:tags.append(policy+'_yaw_only')
                    top=select(v,audio,50)[1];topheat=np.full(cells,np.nan);topheat[top]=audio.max(1)[top]
                    heat=np.stack([heat_values(v.max(1)),heat_values(audio.max(1)),heat_values((.5*standardized(v)+.5*standardized(audio)).max(1)),heat_values(topheat)])
                    name=f'{scene}_{q["sample_index"]}.bin';heat.tofile(target/'heatmaps'/name)
                    result.update({k:old[k] for k in ('scene_id','sample_index','gt_pose_xzyaw','image_path','existing_QA_flags','predicted_visual_rays','predicted_visual_scales','gt_nearest_fp_rays')})
                    if old['existing_QA_flags']:tags.append('existing_geometry_QA_flag')
                    result.update(heatmap='heatmaps/'+name,tags=tags,handcrafted_method=method,
                        acoustic_tied_best_candidates=int((row==row.min()).sum()))
                    records[method].append(result)
                atomic_json(a.report/'status.json',dict(status='running',scene=scene,method=method,completed_queries=min(first+len(values),len(queries)),total_queries=len(queries),updated_unix=time.time()))
            del bank_tensor,candidate
            print('scored',scene,method,len(queries),flush=True)
    summaries={}
    for method,rs in records.items():
        table=summarize(rs);scenes={s:summarize([r for r in rs if r['scene_id']==s]) for s in banks}
        summaries[method]=dict(status='complete',micro=table,per_scene=scenes,queries=len(rs),scenes=len(scenes),
            interpretations=[f'{LABELS[method]}: same1472queries/candidategrid; no test-fit parameters.'],
            paired_pose_bootstrap=paired_intervals(rs),tags={tag:sum(tag in r['tags'] for r in rs) for tag in sorted({tag for r in rs for tag in r['tags']})},
            limitations=learned['limitations']+['Method parameters frozen from exploratory VAL; report all three methods, no test-based selection.'],
            acoustic_tied_queries=sum(r['acoustic_tied_best_candidates']>1 for r in rs),
            failure_analysis=dict(topk_exclusions=sum(not r['diagnostics']['topk_oracle_1m'] for r in rs)))
        target=a.report/method
        atomic_json(target/'summary.json',summaries[method]);atomic_json(target/'records.json',rs);atomic_json(target/'banks.json',banks)
        text=(Path(__file__).with_name('templates')/'pilot_report.html').read_text()
        text=text.replace('HSSD acoustic pilot — 동일 후보 공간 비교',LABELS[method]+' — 동일 Test 후보 공간 비교')
        text=text.replace('acoustic: pilot best checkpoint.',f'acoustic: {LABELS[method]} (학습 없음).')
        text=text.replace("'rgb/'","'../assets/rgb/'").replace("'maps/'","'../assets/maps/'")
        (target/'index.html').write_text(text)
    build_overview(a,learned,summaries,numeric_checks)


def build_overview(a,learned,summaries,numeric_checks):
    import html
    parent=Path('/learned_report')
    assets=a.report/'assets'
    for name in ('rgb','maps'):
        shutil.copytree(parent/name,assets/name,dirs_exist_ok=True)
    all_results=dict(learned=learned,**summaries)
    rows=[]
    for algorithm,result in all_results.items():
        for scope,table_ in [('micro',result['micro']),*result['per_scene'].items()]:
            for policy,metrics in table_.items():rows.append(dict(algorithm=algorithm,scope=scope,policy=policy,**metrics))
    with (a.report/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='complete',queries=1472,scenes=3,methods={m:dict(micro=v['micro'],per_scene=v['per_scene']) for m,v in all_results.items()},
        numerical_checks=numeric_checks,contract=read(a.root/REL/'contract.json'),no_test_fitting=True)
    atomic_json(a.report/'comparison.json',result)
    def table(policy):
        body=''
        for name,data in all_results.items():
            m=data['micro'][policy]
            body+=f'<tr><td>{LABELS[name]}</td>'+''.join(f'<td>{m[k]*100:.2f}%</td>' for k in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg'))+f'<td>{m["median_xy_m"]:.2f}</td><td>{m["median_yaw_deg"]:.1f}°</td></tr>'
        return '<table><tr><th>Acoustic method</th><th>R@0.1m</th><th>R@0.5m</th><th>R@1m</th><th>R@1m·30°</th><th>Median XY(m)</th><th>Median yaw</th></tr>'+body+'</table>'
    links=' · '.join(f'<a href="{name}/">{LABELS[name]} 쿼리·점수 지도</a>' for name in METHODS)
    text='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test learned vs hand-crafted acoustic</title><style>body{font:16px system-ui;margin:30px;background:#f3f6fa;color:#172b45}section{background:white;padding:20px;margin:20px 0;border-radius:10px;overflow:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style><h1>동일 Test: 학습 모델 vs Hand-crafted acoustic</h1><p>3 scenes / 1,472 queries / 0.2m·10° 동일 후보군. 수정된 압출 FP RIR과 furnished observed RIR 재사용. 추가 RIR·학습·test 튜닝 없음. 학습 모델은 VAL로 선택한 epoch5. STFT/coherence/envelope는 앞선 VAL과 동일한 정의·파라미터.</p>'
    for policy,label in [('acoustic_only','Acoustic-only'),('equal_weight_fusion','1:1 fusion'),('visual_topk_acoustic_rerank','Visual top-K, K=50')]:text+='<section><h2>'+label+'</h2>'+table(policy)+'</section>'
    text+='<section><h2>쿼리별 검토</h2><p>'+links+'</p><p><a href="../corrected-test/">학습 모델 쿼리·점수 지도</a> · <a href="metrics.csv">전체 CSV</a> · <a href="comparison.json">Scene별 전체 수치·규약·검증</a></p><p>입력 특징은 float32, 거리는 float64로 계산했고 SciPy와 수치 대조했습니다. 강한 peak 기준으로 정렬하는 legacy STFT와 고정 시계의 coherence/envelope를 동일 정렬 방식인 것처럼 해석하지 않습니다. 같은 test에 대해 여러 방법을 보고한 탐색적 비교이며, 최고값을 독립적인 최종 일반화 성능으로 해석하지 않습니다.</p></section></html>'
    (a.report/'index.html').write_text(text)
    atomic_json(a.report/'status.json',dict(status='complete',results_available=True,queries=1472,updated_unix=time.time()))
    atomic_json(a.root/REL/'evaluation_complete.json',dict(status='complete',report=str(a.report),comparison_sha256=sha(a.report/'comparison.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','extract_cache','evaluate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--report',type=Path);p.add_argument('--vision-hssd',type=Path);p.add_argument('--vision-eval',type=Path)
    p.add_argument('--workers',type=int,default=64);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    try:
        if a.mode=='prepare':prepare(a.root,a.source)
        else:globals()[a.mode](a)
    except Exception:
        target=a.report if a.mode=='evaluate' else a.root/REL
        atomic_json(target/'status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()))
        raise
