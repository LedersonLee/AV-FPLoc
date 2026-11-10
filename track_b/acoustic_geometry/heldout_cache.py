"""Observation/map cache for pre-frozen evaluation; never fits a model."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .prepare_targets import read, sha
from .features import features
from track_b.data.geometry import canonical_rir, candidate_bank
from track_b.data.rays import boundary_rays
from track_b.models.raw_observations import audio_features, visual_features


def main():
    p=argparse.ArgumentParser()
    for key in ['root','study','scale','targets','freeze','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pairs',type=Path);p.add_argument('--source',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--part',choices=['development_holdout','official_test'],required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    frozen=read(a.freeze);scenes=frozen[a.part]
    for c in frozen['checkpoints']:assert sha(a.study/c['run']/c['model']/'model.pt')==c['sha256']
    targets={r['scene']:r for r in read(a.targets/'manifest.json')}
    records={};upstream=None
    if a.pairs:
        assert (a.pairs/'COMPLETE').exists()
        for r in read(a.pairs/'manifest.json'):records[(r['scene'],r['variant'],r['query_index'])]=r
        fp=read(a.scale/'features_v1/protocol.json');assert sha(a.checkpoint)==fp['checkpoint_sha256']
        for file,digest in fp['source_hashes'].items():assert sha(a.source/file)==digest
        sys.path.insert(0,str(a.source));from depth_anything_v2.dpt import DepthAnythingV2
        upstream=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024])
        upstream.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));upstream.cuda().eval().requires_grad_(False)
    else:
        assert a.part=='development_holdout'
        cm={r['scene']:r for r in read(a.scale/'cache_v1/manifest.json')}
        vm={r['scene']:r for r in read(a.scale/'features_v1/manifest.json')}
    protocol=dict(code_sha=a.code_sha,part=a.part,scenes=scenes,freeze_sha256=sha(a.freeze),targets_manifest_sha256=sha(a.targets/'manifest.json'),pairs_manifest_sha256=sha(a.pairs/'manifest.json') if a.pairs else None,inference='RGB/body-aligned RIR/known map only; GT and structural depth are evaluation-only',upstream_checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest=[]
    for scene in scenes:
        start=time.perf_counter();tp=a.targets/f'{scene}.npz';assert sha(tp)==targets[scene]['target_sha256'];t=np.load(tp)
        refs=read(a.targets/f'{scene}_queries.json');assert len(refs)==frozen['queries_per_scene']
        if upstream:
            meta=read(a.root/f'2DFP/{scene}/floorplan_metadata.json');free=np.array(Image.open(a.root/f'2DFP/{scene}/map.png'))==255
            spacing=meta['map_spacing_m'];origin=[meta['world_bounds_m'][k] for k in ['x_min','z_min']]
            poses,_=candidate_bank(free,spacing,origin,.25,36);ranges,hits=boundary_rays(free,spacing,origin,poses[::36,:2],36)
            candidates=np.column_stack([poses,np.log1p(ranges)/math.log(21),hits]).astype(np.float32)
            stride=round(.1/spacing);small=free[::stride,::stride]
            s=dict(scene=scene,candidates=torch.from_numpy(candidates),floorplan=torch.tensor(np.stack([small,~small]),dtype=torch.float32)[None],origin=torch.tensor(origin,dtype=torch.float32),spacing=spacing*stride,hfov_deg=106.2602)
        else:
            cp=a.scale/'cache_v1'/f'{scene}.pt';vp=a.scale/'features_v1'/f'{scene}.pt'
            assert sha(cp)==cm[scene]['cache_sha256'] and sha(vp)==vm[scene]['sha256']
            c=torch.load(cp,map_location='cpu',weights_only=True);v=torch.load(vp,map_location='cpu',weights_only=True)
            assert c['query_ids']==v['query_ids']==[q['query_id'] for q in refs]
            torch.testing.assert_close(c['gt'].double(),torch.from_numpy(t['poses']),atol=2e-6,rtol=1e-6)
            s={k:c[k] for k in ['scene','candidates','floorplan','origin','spacing','hfov_deg']};s['visual']=v['visual_features']
        s.update(gt=torch.from_numpy(t['poses']).float(),poses=torch.from_numpy(t['poses']),depth=torch.from_numpy(t['depth']),mask=torch.from_numpy(t['mask']),query_ids=[q['query_id'] for q in refs],conditions={})
        checks=[];visual=[]
        for condition in ['furnished','stage_only'] if upstream else ['historical_furnished']:
            echo=[];spatial=[];legacy=[]
            for i,q in enumerate(refs):
                if upstream:
                    r=records[(scene,condition,i)];path=a.pairs/r['file'];assert sha(path)==r['sha256']
                    assert r['stage_sha256']==targets[scene]['stage_sha256'] and r['query_id']==q['query_id']
                    assert r['world_yaw_deg']==q['world_yaw_deg'] and r['base_acoustic_yaw_deg']==q['base_acoustic_yaw_deg']
                    assert np.allclose(r['centre_xyz_m'],q['centre_xyz_m']) and r['source_xyz_m']==r['centre_xyz_m']
                else:
                    path=a.root/q['rir_path'];side=read(path.with_suffix('.json'));assert sha(path)==side['rir_sha256']
                    assert side['role']=='Observed' and side['acoustics']['sample_rate_hz']==48000
                    assert np.allclose(side['centre_xyz_m'],q['centre_xyz_m']) and np.allclose(side['source_xyz_m'],q['centre_xyz_m'])
                    assert side['base_acoustic_yaw_deg']==q['base_acoustic_yaw_deg']
                body=canonical_rir(np.load(path),q['world_yaw_deg'],q['base_acoustic_yaw_deg']).copy()
                e,g=features(torch.from_numpy(body)[None]);echo.append(e[0].half());spatial.append(g[0].half());legacy.append(audio_features(body)[0])
                checks.append(dict(condition=condition,query_id=q['query_id'],rir_sha256=sha(path)))
                if upstream and condition=='furnished':
                    rgb=a.root/q['rgb_path'];image=cv2.imread(str(rgb));assert image is not None
                    visual.append(visual_features(upstream,image)[0].cpu());checks[-1]['rgb_sha256']=sha(rgb)
            s['conditions'][condition]=dict(echo=torch.stack(echo),spatial=torch.stack(spatial),spectrum=torch.stack(legacy))
        if upstream:s['visual']=torch.stack(visual)
        else:torch.testing.assert_close(s['conditions']['historical_furnished']['spectrum'],c['spectrum'],atol=0,rtol=0)
        dest=a.out/f'{scene}.pt';torch.save(s,dest);(a.out/f'{scene}_inputs.json').write_text(json.dumps(checks,indent=2))
        row=dict(scene=scene,sha256=sha(dest),seconds=time.perf_counter()-start,queries=len(refs),candidates=len(s['candidates']))
        manifest.append(row);(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(row),flush=True)
    (a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()
