"""Require every declared condition; scene-paired comparisons without selection."""
import argparse,json,hashlib,csv,math
from pathlib import Path
import numpy as np
import torch
from track_b.eval.extended_metrics import reliability_statistics

CONDITIONS=['normal','occlusion25','occlusion50','occlusion75','brightness','audio_shuffle','audio_noise20']
METRICS=['joint','r1','fine','strict_05_15','strict_02_10','raw5','raw10','raw20','diverse20','nll','normalized_entropy']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2))

def paired_scene(left,right,scenes,metric):
    def key(r):return r['seed'],r['scene'],r['query']
    a={key(r):r for r in left};b={key(r):r for r in right}
    assert len(a)==len(left) and len(b)==len(right) and a.keys()==b.keys()
    delta=np.array([np.mean([a[k][metric]-b[k][metric] for k in a if k[1]==sid]) for sid in scenes])
    assert np.isfinite(delta).all()
    factor=1 if metric in ['nll','normalized_entropy'] else 100
    rng=np.random.default_rng(20261109);boot=delta[rng.integers(len(scenes),size=(10000,len(scenes)))].mean(1)*factor
    return dict(delta=float(delta.mean()*factor),ci95=np.quantile(boot,[.025,.975]).tolist(),scene_deltas=(delta*factor).tolist(),units='native' if factor==1 else 'percentage points',sampling='paired scenes; queries and seeds averaged within scene')


def ray_stats(records):
    mean=np.stack([x['mean'].numpy() for x in records]);target=np.stack([x['target'].numpy() for x in records]);b=np.stack([x['scale'].numpy() for x in records]);visible=np.stack([x['visible'].numpy() for x in records]).astype(bool);error=np.abs(target-mean)
    masks={'visible':visible,'outside_fov':~visible,'short_le5m':visible&(target<=math.log(6)/math.log(21)),'long_gt5m':visible&(target>math.log(6)/math.log(21))}
    for k in range(16):
        m=np.zeros_like(visible);m[:,k]=True;masks[f'direction_{k}']=m
    out=[]
    for name,mask in masks.items():
        ee=error[mask];bb=b[mask]
        if not len(ee):out.append(dict(stratum=name,count=0));continue
        correlation=float(np.corrcoef(ee,bb)[0,1]) if np.ptp(ee)>1e-12 and np.ptp(bb)>1e-12 else None
        bins=[]
        # Equal-frequency diagnostic; no parameters are calibrated/refit.
        chunks=np.array_split(np.argsort(bb,kind='stable'),10) if np.ptp(bb)>1e-12 else [np.arange(len(bb))]
        for ids in chunks:
            if len(ids):bins.append(dict(count=len(ids),mean_scale=float(bb[ids].mean()),mean_error=float(ee[ids].mean())))
        out.append(dict(stratum=name,count=len(ee),mae=float(ee.mean()),nll=float((ee/bb+np.log(2*bb)).mean()),mean_scale=float(bb.mean()),pearson_scale_error=correlation,coverage={str(q):float((ee<=-bb*np.log(1-q)).mean()) for q in [.5,.8,.9,.95]},scale_error_bins=bins,scope='Outside-FoV V and individual invisible directions are unsupported diagnostics, not likelihood contributions'))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
    required=[a.root/f'eval_{c}_v1' for c in CONDITIONS]
    # Fail before creating output if any full result is missing/incomplete.
    for r in required:
        for f in ['protocol.json','predictions.json','summary.json','ray_records.pt','validation.json','timing.json','field_spreads.json']:assert (r/f).is_file(),f'Await complete declared evaluations: {r.name}/{f}'
    protocols=[read(r/'protocol.json') for r in required];assert [p['condition'] for p in protocols]==CONDITIONS;scenes=protocols[0]['scenes'];assert len(scenes)==19 and len(set(scenes))==19
    assert all(not p['smoke'] and p['scenes']==scenes and p['seeds']==[42,43,44] and p['queries_per_scene']==64 for p in protocols)
    assert all(p['head_protocol_sha256']==protocols[0]['head_protocol_sha256'] and p['declared_protocol_sha256']==protocols[0]['declared_protocol_sha256'] for p in protocols)
    summaries=[];contrasts=[];transfers=[];secondary=[];ray_summary=[];sources={};cohort=None;normal_groups=None
    for condition,r in zip(CONDITIONS,required):
        rows=read(r/'predictions.json');expected=98496 if condition=='normal' else 54720
        assert len(rows)==expected==read(r/'validation.json')['records']
        keys=[(x['arm'],x['method'],x['policy'],x['seed'],x['scene'],x['query']) for x in rows];assert len(set(keys))==len(keys)
        groups={}
        for x in rows:groups.setdefault((x['arm'],x['method'],x['policy']),[]).append(x)
        expected_groups={(arm,m,'primary') for arm in ['U0','U1','U2','U3','U4'] for m in ['visual','acoustic','fixed']}
        if condition=='normal':expected_groups|={(arm,m,'original_scales') for arm in ['U1','U2','U3','U4'] for m in ['visual','acoustic','fixed']}
        assert set(groups)==expected_groups
        saved_summary=read(r/'summary.json')
        saved={(x['arm'],x['method'],x['policy']):x for x in saved_summary}
        assert len(saved)==len(saved_summary) and set(saved)==expected_groups
        for key,rr in groups.items():
            kk={(x['seed'],x['scene'],x['query']) for x in rr};assert len(rr)==len(kk)==3648
            if cohort is None:cohort=kk
            assert kk==cohort and all(sum(x['scene']==s and x['seed']==seed for x in rr)==64 for s in scenes for seed in [42,43,44])
            means={m:float(np.mean([x[m] for x in rr])) for m in METRICS}
            reliability=reliability_statistics(rr)
            assert saved[key]['queries']==len(rr)
            for metric,value in means.items():
                assert np.isfinite(value) and abs(value-saved[key]['mean'][metric])<1e-9,(condition,key,metric)
            assert reliability==saved[key]['reliability'],(condition,key,'reliability replay')
            summaries.append(dict(condition=condition,arm=key[0],method=key[1],policy=key[2],queries=len(rr),**means,reliability=reliability))
        if condition=='normal':
            normal_groups=groups
            for arm in ['U1','U2','U3','U4']:
                for method in ['visual','acoustic','fixed']:
                    for metric in METRICS:
                        secondary.append(dict(arm=arm,method=method,metric=metric,left='original_scales',right='primary',**paired_scene(groups[arm,method,'original_scales'],groups[arm,method,'primary'],scenes,metric)))
        else:
            for arm in ['U0','U1','U2','U3','U4']:
                for method in ['visual','acoustic','fixed']:
                    for metric in METRICS:
                        transfers.append(dict(condition=condition,arm=arm,method=method,metric=metric,left=condition,right='normal',**paired_scene(groups[arm,method,'primary'],normal_groups[arm,method,'primary'],scenes,metric)))
        for method in ['visual','acoustic','fixed']:
            for control in ['U1','U2','U4','U0']:
                for metric in METRICS:contrasts.append(dict(condition=condition,left='U3',right=control,method=method,metric=metric,**paired_scene(groups['U3',method,'primary'],groups[control,method,'primary'],scenes,metric)))
        for control in ['U0','U3']:
            for metric in ['joint','fine','raw20','diverse20','nll']:contrasts.append(dict(condition=condition,left='U3 fixed',right=control+' acoustic',method='negative_transfer',metric=metric,**paired_scene(groups['U3','fixed','primary'],groups[control,'acoustic','primary'],scenes,metric)))
        rays=torch.load(r/'ray_records.pt',weights_only=True,map_location='cpu');assert len(rays)==29184
        for arm in ['U1','U2','U3','U4']:
            for modality in ['visual','acoustic']:
                rr=[x for x in rays if x['arm']==arm and x['modality']==modality];assert len(rr)==3648 and {(x['seed'],x['scene'],x['query']) for x in rr}==cohort
                ray_summary.append(dict(condition=condition,arm=arm,modality=modality,strata=ray_stats(rr)))
        sources[condition]={f:sha(r/f) for f in ['protocol.json','predictions.json','summary.json','ray_records.pt','validation.json']}
    a.out.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary',summaries),('paired',contrasts),('normal_shift_transfer',transfers),('secondary_scale_policy',secondary),('ray_calibration',ray_summary),('validation',dict(code_sha=a.code_sha,conditions=CONDITIONS,scenes=scenes,query_seed_keys=len(cohort),source_hashes=sources,status='all declared conditions validated',evidence='Exploratory reused Val; uncorrected multiple comparisons'))]:write(a.out/f'{name}.json',value)
    print('COMPLETE',len(summaries),len(contrasts),flush=True)
if __name__=='__main__':main()
