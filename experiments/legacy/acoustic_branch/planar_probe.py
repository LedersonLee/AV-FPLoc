"""First-order planar proxy + matched vertical ablation + reflection timing.

Exploratory VAL only. This is not a high-order/full-wave 2D RIR simulator.
Never overwrites shared maps, training data, or frozen native/extruded RIRs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy import signal,ndimage
from scipy.spatial.distance import cdist

from .contracts import atomic_json,immutable_json,sha
from .data import read,load_manifest
from .features import extract
from .planar_paths import segments_from_mask,render
from .stage_comparison import stage_path


@lru_cache(maxsize=3)
def geometry(root,scene):
    root=Path(root)
    meta=read(root/f'2DFP/{scene}/floorplan_metadata.json')
    free=np.asarray(Image.open(root/f'2DFP/{scene}/map.png').convert('L'))==255
    height=read(root/f'RIR/Geometry/{scene}/floorplan.json')
    walls=segments_from_mask(free,meta['map_spacing_m'],meta['world_bounds_m']['x_min'],meta['world_bounds_m']['z_min'])
    return walls,height['floor_y_m'],height['ceiling_y_m']


def proxy_path(output,kind,original):
    return Path(output)/'rirs'/kind/Path(original).relative_to('RIR/FP/val')


def render_task(task):
    root,output,path,ref,contract=task
    for kind in ('planar_first','vertical_first'):
        p=proxy_path(output,kind,path)
        if not p.exists() or not p.with_suffix('.json').exists():break
        m=read(p.with_suffix('.json'))
        if m['contract_sha256']!=contract or m['sha256']!=sha(p):raise ValueError('Proxy output changed')
    else:return path
    walls,floor,ceiling=geometry(root,Path(path).parts[3])
    base=int(Path(path).stem.rsplit('_',1)[1])
    planar,vertical,paths=render(ref['xyz'],base,walls,floor,ceiling)
    for kind,array in [('planar_first',planar),('vertical_first',vertical)]:
        p=proxy_path(output,kind,path);p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix('.tmp.npy');np.save(tmp,array,allow_pickle=False);tmp.replace(p)
        atomic_json(p.with_suffix('.json'),dict(sha256=sha(p),contract_sha256=contract,
            reference=ref,kind=kind,paths=paths,direct_sample=7,global_filter_delay_samples=0))
    return path


def arrival_features(rir,start_ms=2.):
    """Fixed-clock low-band envelope and sparse peaks, no per-record shifting."""
    fs=48000;first=7+round(start_ms*48);last=7+6144
    x=np.zeros((6,last),float)
    n=min(last,rir.shape[-1]);x[:,first:n]=rir[:,first:n]
    # Remove direct/guard samples BEFORE filtering, to avoid treating the
    # direct pulse's low-frequency filter tail as a wall reflection.
    x=np.pad(x,((0,0),(1024,1024)))
    sos=signal.butter(3,[500,1500],btype='bandpass',fs=fs,output='sos')
    x=signal.sosfiltfilt(sos,x,axis=-1)
    envelope=abs(signal.hilbert(x,axis=-1))[:,1024+first:1024+last]
    envelope=ndimage.gaussian_filter1d(envelope,12,axis=-1)
    frames=envelope.shape[-1]//24
    envelope=envelope[:,:frames*24].reshape(6,frames,24).mean(-1)
    maximum=envelope.max(-1,keepdims=True)
    norm=np.divide(envelope,maximum,out=np.zeros_like(envelope),where=maximum>1e-15)
    events=np.zeros_like(norm)
    for i,row in enumerate(norm):
        peaks,_=signal.find_peaks(row,height=.1,distance=2,prominence=.05)
        events[i,peaks]=1
    dilated=ndimage.maximum_filter1d(norm,3,axis=-1,mode='constant')
    return norm.astype(np.float32),events.astype(np.float32),dilated.astype(np.float32)


def feature(rir,config,m,legacy):
    stft=m.featurise(m.band_energy(rir,legacy),legacy).astype(np.float32)
    try:coh=extract(rir,config)[1]
    except ValueError as e:
        if 'No reflection energy' not in str(e):raise
        coh=np.zeros((2,6,2,19),np.float32)
    env,event,dilated=arrival_features(rir)
    residual=rir.astype(np.float64)-rir.mean(axis=0,dtype=np.float64)
    early=dict(config,start_ms=.35,early_end_ms=20)
    def coherence(array,settings):
        if not np.all(np.any(array!=0,axis=-1)):return np.zeros((2,6,2,19),np.float32)
        try:return extract(array,settings)[1]
        except ValueError as e:
            if 'No reflection energy' not in str(e):raise
            return np.zeros((2,6,2,19),np.float32)
    common_env,common_event,common_dilated=arrival_features(residual)
    return dict(stft=stft,coherence=coh,envelope=env,event=event,dilated=dilated,
        common_coherence=coherence(residual,config),early_coherence=coherence(rir,early),
        early_common_coherence=coherence(residual,early),common_envelope=common_env,
        common_event=common_event,common_dilated=common_dilated)


def rolled(f,roll):
    return {k:(np.roll(v.reshape(6,3,-1),roll,0).reshape(v.shape) if k=='stft' else
               np.roll(v,roll,1 if 'coherence' in k else 0)) for k,v in f.items()}


def distances(ob,bank,method):
    key=dict(STFT_L1='stft',coherence_L2='coherence',arrival_L1='envelope',
        common_coherence_L2='common_coherence',early_coherence_L2='early_coherence',
        early_common_coherence_L2='early_common_coherence',common_arrival_L1='common_envelope').get(method)
    if key:
        return cdist(np.asarray([f[key].ravel() for f in ob]),np.asarray([f[key].ravel() for f in bank]),
            metric='sqeuclidean' if 'coherence' in method else 'cityblock')
    prefix='common_' if method.startswith('common_') else ''
    o=np.asarray([f[prefix+'dilated'].ravel() for f in ob],dtype=float)
    c=np.asarray([f[prefix+'event'].ravel() for f in bank],dtype=float)
    count=c.sum(-1,keepdims=True)
    # Candidate arrival coverage, not an unconstrained energy one-sided L1.
    # Empty candidates receive worst score rather than a free perfect match.
    c=np.divide(c,count,out=np.zeros_like(c),where=count>0)
    return np.clip(1-o@c.T,0,1)


def summarize(rows):
    return dict(queries=len(rows),sampled_top1=float(np.mean([r['sampled_credit'] for r in rows])),
        **{f'recall_{t:g}m':float(np.mean([r['xy_error_m']<=t for r in rows])) for t in (.1,.5,1)},
        recall_1m_30deg=float(np.mean([r['xy_error_m']<=1 and r['yaw_error_deg']<=30 for r in rows])),
        median_xy_error_m=float(np.median([r['xy_error_m'] for r in rows])),
        median_yaw_error_deg=float(np.median([r['yaw_error_deg'] for r in rows])),
        tied_queries=sum(r['ties']>1 for r in rows),
        gt_event_coverage=float(np.mean([r['gt_event_coverage'] for r in rows])))


def run(a):
    config=read(a.config);queries,digest=load_manifest(a.root,'val',config)
    refs={r['path']:r for q in queries for r in q['candidates']}
    maps={str(p.relative_to(a.root)):sha(p) for scene in sorted({q['scene_id'] for q in queries})
        for p in (a.root/f'2DFP/{scene}/map.png',a.root/f'2DFP/{scene}/floorplan_metadata.json',a.root/f'RIR/Geometry/{scene}/floorplan.json')}
    contract=dict(version='planar_first_order_v1',test_accessed=False,manifest_sha256=digest,
        config_sha256=sha(a.config),maps=maps,code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('planar_paths.py'))},
        solver='specular first-order horizontal image sources; no high-order/scattering/diffraction',
        amplitude='reflection .8; spherical 1/(4pi*d) on planar paths, NOT cylindrical wave acoustics',
        vertical_control='identical solver + floor/ceiling single-bounce arrivals',
        timing='48kHz; c343m/s; radius .05m; fixed origin, direct~sample7; sinc81 delay0',
        matching='500-1500Hz fixed-clock envelope, .5ms bins, +/- .5ms tolerance; no GT alignment',
        selection='exploratory full pilot VAL, no test, no learning; report every configuration')
    if a.reuse_proxy:
        previous=read(a.reuse_proxy/'contract.json')
        if previous['maps']!=maps or previous['manifest_sha256']!=digest or previous['code_sha256']['planar_paths.py']!=sha(Path(__file__).with_name('planar_paths.py')):
            raise ValueError('Cannot reuse different proxy physics/maps/poses')
        contract['reused_proxy_contract_sha256']=sha(a.reuse_proxy/'contract.json')
    contract['common_mode_test']='subtract six-mic mean at each sample; calibrated central source, same horizontal ring height. Coherence 2ms/50ms and .35ms/20ms factorial.'
    immutable_json(a.output/'contract.json',contract)
    raw_proxy=a.reuse_proxy or a.output
    if a.reuse_proxy:
        for p,r in refs.items():
            for kind in ('planar_first','vertical_first'):
                path=proxy_path(raw_proxy,kind,p);meta=read(path.with_suffix('.json'))
                if meta['contract_sha256']!=sha(a.reuse_proxy/'contract.json') or sha(path)!=meta['sha256'] or meta['reference']['xyz']!=r['xyz']:
                    raise ValueError('Reused proxy raw verification failed')
    else:
        tasks=[(str(a.root),str(a.output),p,r,sha(a.output/'contract.json')) for p,r in refs.items()]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,_ in enumerate(pool.map(render_task,tasks),1):
                if i%32==0 or i==len(tasks):
                    atomic_json(a.output/'status.json',dict(status='running',stage='planar_render',completed=i,total=len(tasks),updated_unix=time.time()))
    spec=importlib.util.spec_from_file_location('grid_planar',a.source)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    legacy=m.GridScoreConfig(sample_rate_hz=48000,direct_guard_samples=96,usable_samples=6144,nfft=256,hop=64,
        bands=((0,500),(500,1500),(1500,4000)),normalise='sum',distance='symmetric',shape_feature=True,feature='stft_band')
    native=a.root/'AcousticContrastive/diagnostics/stage_vs_extruded_v1'
    allrefs={r['path']:r for q in queries for r in [q['observed']]+q['candidates']}
    jobs=[('original',p) for p in allrefs]+[(k,p) for k in ('native_stage','planar_first','vertical_first') for p in refs]
    def load(job):
        kind,p=job
        path=a.root/p if kind=='original' else stage_path(native,p) if kind=='native_stage' else proxy_path(raw_proxy,kind,p)
        meta=read(path.with_suffix('.json'))
        if sha(path)!=meta['rir_sha256' if kind=='original' else 'sha256']:raise ValueError('Raw SHA mismatch')
        return job,feature(np.load(path,allow_pickle=False),config['features'],m,legacy)
    atomic_json(a.output/'status.json',dict(status='running',stage='reflection_features',arrays=len(jobs),updated_unix=time.time()))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:cache=dict(pool.map(load,jobs))
    result=dict(contract_sha256=sha(a.output/'contract.json'),source_sha256=sha(a.source),test_accessed=False,methods={},
        scope='captured-pose VAL banks; primary planar proxy is FIRST ORDER, NOT full 2D RIR simulation')
    for kind,label in [('original','extruded_RLR'),('native_stage','native_RLR'),('planar_first','planar_first'),('vertical_first','vertical_first')]:
        methods={name:[] for name in ('STFT_L1','coherence_L2','arrival_L1','arrival_partial',
            'common_coherence_L2','early_coherence_L2','early_common_coherence_L2','common_arrival_L1','common_arrival_partial')}
        for scene in sorted({q['scene_id'] for q in queries}):
            scenequeries=[q for q in queries if q['scene_id']==scene]
            # Fixed hash order makes exact ties independent of GT/query order.
            ordered=sorted(scenequeries,key=lambda q:hashlib.sha256(f'{scene}:{q["sample_index"]}'.encode()).hexdigest())
            candidates=[q['candidates'][0] for q in ordered]
            bank=[rolled(cache[kind,r['path']],r['roll']) for r in candidates]
            observed=[rolled(cache['original',q['observed']['path']],q['observed']['roll']) for q in scenequeries]
            for method in methods:
                d=distances(observed,bank,method)
                coverage=1-distances(observed,bank,'arrival_partial')
                gt_indices={q['sample_index']:i for i,q in enumerate(ordered)}
                for i,q in enumerate(scenequeries):
                    j=int(d[i].argmin());prediction=candidates[j]
                    pool=[rolled(cache[kind,r['path']],r['roll']) for r in q['candidates']]
                    ds=distances([observed[i]],pool,method)[0];win=ds==ds.min()
                    xy=float(np.linalg.norm((np.asarray(q['observed']['xyz'])-prediction['xyz'])[[0,2]]))
                    yaw=abs((q['observed']['yaw']-prediction['yaw']+180)%360-180)
                    methods[method].append(dict(scene=scene,sample_index=q['sample_index'],xyz=q['observed']['xyz'],
                        gt_yaw=q['observed']['yaw'],image_path=q['image_path'],sampled_credit=float(win[0]/win.sum()),
                        xy_error_m=xy,yaw_error_deg=yaw,predicted_xyz=prediction['xyz'],predicted_yaw=prediction['yaw'],
                        ties=int((d[i]==d[i].min()).sum()),gt_event_coverage=float(coverage[i,gt_indices[q['sample_index']]])))
        result['methods'][label]={name:dict(summary=summarize(rows),
            per_scene={s:summarize([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})},records=rows)
            for name,rows in methods.items()}
        atomic_json(a.output/'results.json',result)
        atomic_json(a.output/'status.json',dict(status='running',stage='evaluated_'+label,updated_unix=time.time()))
        print(label,{name:v['summary'] for name,v in result['methods'][label].items()},flush=True)
    atomic_json(a.output/'status.json',dict(status='complete',test_accessed=False,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','output','source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--reuse-proxy',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    try:run(a)
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='failed',error=repr(e)));raise
