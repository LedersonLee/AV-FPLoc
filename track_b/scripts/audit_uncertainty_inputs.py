"""Read-only integrity gate for Study30's specified independent F1 inputs."""
import argparse, hashlib, json
from pathlib import Path

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);a=p.parse_args()
    r=a.root;cache=r/'cache_v1';features=r/'features_v1';reference=r/'n112_budget1_evaluation_v1'
    cp=read(cache/'protocol.json');rp=read(reference/'protocol.json');fp=read(features/'protocol.json')
    assert len(cp['scenes']['train'])==112 and len(cp['scenes']['val'])==19
    assert not set(cp['scenes']['train'])&set(cp['scenes']['val'])
    assert cp['queries_per_scene']==64 and cp['grid_m']==.25 and cp['yaw_bins']==36 and cp['ray_bins']==16
    assert cp['sample_rate_hz']==48000 and cp['rir_samples']==16384 and cp['spectral_channels']==18
    assert rp['scene_ids']==cp['scenes']['val'] and rp['seeds']==[42,43,44]
    assert rp['fixed_only'] and rp['fixed']==dict(visual_weight=.85,temperature=1.)
    assert rp['cache_manifest_sha256']==sha(cache/'manifest.json')
    assert rp['evaluation_visual_cache_protocol_sha256']==sha(features/'protocol.json') and fp['data']==cp
    checkpoints={}
    for mode,stage in [('visual','visual_joint'),('acoustic','acoustic')]:
        run=r/'scaling_fits_v1'/f'n112_budget1_{stage}_attempt1';prot=read(run/'protocol.json')
        assert prot['data']['scenes']==cp['scenes'] and prot['seeds']==rp['seeds']
        assert prot['cache_protocol_sha256']==sha(cache/'protocol.json')
        for seed in rp['seeds']:
            path=run/f'{mode}_{seed}.pt';digest=sha(path);assert digest==rp['checkpoint_hashes'][f'{mode}_{seed}']
            checkpoints[f'{mode}_{seed}']=dict(path=str(path),sha256=digest)
    ids=set(cp['scenes']['train']+cp['scenes']['val']);checks=[]
    for folder,key in [(cache,'cache_sha256'),(features,'sha256')]:
        manifest=read(folder/'manifest.json');assert len(manifest)==131 and {x['scene'] for x in manifest}==ids
        for x in manifest:
            sid=x['scene'];assert x['queries']==64 and sha(folder/f'{sid}.pt')==x[key]
            if folder==features:assert sha(cache/f'{sid}_inputs.json')==x['input_manifest_sha256']
        checks.append(dict(path=str(folder),scenes=len(manifest),queries=8384,hashes_match=True))
    rows=read(reference/'predictions.json');keys=[(x['method'],x['seed'],x['scene'],x['query']) for x in rows]
    assert len(keys)==len(set(keys))==14592
    summary=read(reference/'summary.json')['summary']
    for method in ['visual','acoustic','fixed','fixed_audio_shuffle']:
        group=[x for x in rows if x['method']==method];assert len(group)==3648
        for metric in ['joint','fine','raw20','diverse20','nll']:
            assert abs(sum(x[metric] for x in group)/len(group)-summary[method][metric])<1e-9
    a.out.mkdir(parents=True,exist_ok=False)
    result=dict(code_sha=a.code_sha,status='input integrity passed; GPU replay still required',checkpoints=checkpoints,cache_checks=checks,reference_summary=summary,protocol_hashes={str(x):sha(x) for x in [cache/'protocol.json',features/'protocol.json',reference/'protocol.json']},reference_predictions_sha256=sha(reference/'predictions.json'),test_accessed=False,scope='Cached artifacts and per-query metric consistency; not raw-file rehash, full-field replay, or independent test')
    (a.out/'validation.json').write_text(json.dumps(result,indent=2));print('PASS',flush=True)
if __name__=='__main__':main()
