"""Frozen-model furniture comparison with fresh-render and repeat controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .prepare_targets import read,sha
from .features import features
from .models import GeometryNet
from .summarize import paired
from track_b.data.geometry import canonical_rir
from track_b.models.raw_observations import audio_features


def normalized_waveform(body):
    x=np.zeros((6,16384),np.float32);n=min(16384,body.shape[-1]);x[:,:n]=body[:,:n]
    return x/max(float(np.sqrt(np.mean(x*x))),1e-8)


def metrics(mean,logits,y,mask,centres):
    e=mean-y;r=dict(mae=float(e.abs()[mask].mean()),rmse=float(e.square()[mask].mean().sqrt()),angular_sd_m=float(mean.std(-1,unbiased=False).mean()))
    if logits is not None:
        lp=logits.log_softmax(-1);p=lp.exp();label=(y*4).long().clamp(0,79)
        r['nll']=float(-lp.gather(-1,label[...,None])[...,0][mask].mean())
        cdf=p.cumsum(-1);lo=(cdf<.05).sum(-1).clamp_max(79)/4;hi=((cdf<.95).sum(-1).clamp_max(79)+1)/4
        r['coverage90']=float(((y>=lo)&(y<=hi))[mask].float().mean());r['width90_m']=float((hi-lo)[mask].mean())
    return r


def main():
    p=argparse.ArgumentParser()
    for k in ['root','study','pairs','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
    assert (a.pairs/'COMPLETE').exists();pc=read(a.pairs/'protocol.json');manifest=read(a.pairs/'manifest.json')
    records={};target_manifest={r['scene']:r for r in read(a.study/'targets_v3/manifest.json')}
    for r in manifest:
        path=a.pairs/r['file'];assert sha(path)==r['sha256']
        assert r['stage_sha256']==target_manifest[r['scene']]['stage_sha256']
        key=(r['scene'],r['variant'],r['query_index']);assert key not in records;records[key]=r
    corpus=[];wave_checks=[]
    for scene in pc['scenes']:
        indices=sorted(k[2] for k in records if k[:2]==(scene,'furnished'))
        assert len(indices)==pc['queries_per_scene']
        z=np.load(a.study/'targets_v3'/f'{scene}.npz');refs=read(a.study/'targets_v3'/f'{scene}_queries.json')
        package=dict(scene=scene,indices=indices,query_ids=[refs[i]['query_id'] for i in indices],
                     depth=torch.from_numpy(z['depth'][indices]),mask=torch.from_numpy(z['mask'][indices]),conditions={})
        normalized={}
        for condition in ['historical_furnished','furnished','stage_only','furnished_repeat']:
            selected=indices[:1] if condition=='furnished_repeat' else indices
            echo=[];spatial=[];legacy=[];norm=[]
            for i in selected:
                q=refs[i]
                if condition=='historical_furnished':
                    path=a.root/q['rir_path'];side=read(path.with_suffix('.json'));assert sha(path)==side['rir_sha256']
                else:
                    r=records[(scene,condition,i)];path=a.pairs/r['file']
                    assert r['query_id']==q['query_id'] and r['world_yaw_deg']==q['world_yaw_deg']
                    assert r['base_acoustic_yaw_deg']==q['base_acoustic_yaw_deg']
                    assert np.allclose(r['centre_xyz_m'],q['centre_xyz_m']) and r['source_xyz_m']==r['centre_xyz_m']
                    assert r['loaded_objects']==0 if condition=='stage_only' else r['loaded_objects']>0
                body=canonical_rir(np.load(path),q['world_yaw_deg'],q['base_acoustic_yaw_deg']).copy()
                e,s=features(torch.from_numpy(body)[None]);echo.append(e[0]);spatial.append(s[0]);legacy.append(audio_features(body)[0]);norm.append(normalized_waveform(body))
            package['conditions'][condition]=dict(echo=torch.stack(echo).half(),spatial=torch.stack(spatial).half(),legacy=torch.stack(legacy))
            normalized[condition]=np.stack(norm)
        for left,right in [('furnished','stage_only'),('historical_furnished','furnished'),('furnished_repeat','furnished')]:
            x=normalized[left];y=normalized[right][:len(x)]
            wave_checks.append(dict(scene=scene,left=left,right=right,normalized_l2=float(np.sqrt(np.mean((x-y)**2))),samples=len(x)))
        torch.save(package,a.out/f'{scene}_features.pt');corpus.append(package)
    rows=[];prediction_changes=[];checks=[]
    for record in read(a.study/'representation_v1/registry.json'):
        checkpoint=a.study/'representation_v1'/record['run']/'model.pt';assert sha(checkpoint)==record['checkpoint_sha256']
        model=GeometryNet(record['architecture'],record['head']);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
        for s in corpus:
            predicted={}
            with torch.no_grad():
                for condition,x in s['conditions'].items():
                    m,l=model(x[record['architecture']].float());n=len(m)
                    values=metrics(m,l,s['depth'][:n],s['mask'][:n],model.centres)
                    rows.append(dict(scene=s['scene'],architecture=record['architecture'],head=record['head'],seed=record['seed'],condition=condition,queries=n,**values))
                    predicted[condition]=m
            for left,right in [('furnished','stage_only'),('historical_furnished','furnished'),('furnished_repeat','furnished')]:
                x=predicted[left];y=predicted[right][:len(x)]
                prediction_changes.append(dict(scene=s['scene'],architecture=record['architecture'],head=record['head'],seed=record['seed'],left=left,right=right,
                    mean_absolute_prediction_change_m=float((x-y).abs().mean())))
        checks.append(dict(run=record['run'],checkpoint_sha256=record['checkpoint_sha256']))
    grouped={}
    for r in rows:
        for metric in ['mae','rmse','angular_sd_m','nll','coverage90','width90_m']:
            if metric in r:grouped.setdefault((r['architecture'],r['head'],r['condition'],metric),{}).setdefault(r['scene'],[]).append(r[metric])
    means={k:{s:float(np.mean(v)) for s,v in d.items()} for k,d in grouped.items()}
    summary=[dict(architecture=k[0],head=k[1],condition=k[2],metric=k[3],**paired(v)) for k,v in means.items()]
    contrasts=[]
    for arch,head in sorted({k[:2] for k in means}):
        for left,right in [('furnished','stage_only'),('historical_furnished','furnished')]:
            for metric in ['mae','rmse','nll']:
                if (arch,head,left,metric) not in means:continue
                l=means[(arch,head,left,metric)];r=means[(arch,head,right,metric)]
                contrasts.append(dict(architecture=arch,head=head,left=left,right=right,metric=metric,**paired({s:l[s]-r[s] for s in l})))
    result=dict(code_sha=a.code_sha,pair_protocol_sha256=sha(a.pairs/'protocol.json'),pair_manifest_sha256=sha(a.pairs/'manifest.json'),
        scenes=len(corpus),pairs=sum(len(s['indices']) for s in corpus),validation='all raw hashes, original target stage hashes, poses, headings, source geometry and checkpoint hashes checked',
        evidence=f"Exploratory{len(corpus)} development families,{pc['queries_per_scene']} furnished-selected poses each; default material simulation; no real-world or furniture-invariance claim",
        summary=summary,contrasts=contrasts,per_scene=rows,prediction_changes=prediction_changes,waveform_checks=wave_checks,checkpoints=checks)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Matched furniture comparison','',result['evidence'],'','| Encoder/head | Historical furnished MAE | Fresh furnished MAE | Stage-only MAE | Fresh furnished − stage-only,95% CI |','|---|---:|---:|---:|---:|']
    lookup={(r['architecture'],r['head'],r['condition'],r['metric']):r for r in summary}
    for arch,head in sorted({k[:2] for k in means}):
        vals=[lookup[(arch,head,c,'mae')]['mean'] for c in ['historical_furnished','furnished','stage_only']]
        delta=next(r for r in contrasts if (r['architecture'],r['head'],r['left'],r['right'],r['metric'])==(arch,head,'furnished','stage_only','mae'))
        lines.append(f"| {arch}/{head} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {delta['mean']:.3f} [{delta['ci95'][0]:.3f},{delta['ci95'][1]:.3f}] |")
    lines+=['','Fresh furnished re-renders and first-query repeated simulations expose renderer variability. Small mean differences are not evidence of furniture invariance, especially when directional reconstruction itself fails.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
