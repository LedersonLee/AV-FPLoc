"""Freeze larger scene-held-out development data; cache observations, not references."""
import argparse,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from track_b.data.geometry import candidate_bank,canonical_rir
from track_b.data.rays import boundary_rays,rays_at_poses
from track_b.models.structure import rir_spectrum


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jread(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--train-scenes',type=int,default=16);ap.add_argument('--val-scenes',type=int,default=6);ap.add_argument('--queries',type=int,default=64);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);split=jread(args.audit/'split_manifest.json')['development'];key=lambda s:hashlib.sha256(s.encode()).hexdigest()
    chosen={'train':sorted(split['train'],key=key)[:args.train_scenes],'val':sorted(split['val'],key=key)[:args.val_scenes]}
    assert not ({s.split('_')[0] for s in chosen['train']}&{s.split('_')[0] for s in chosen['val']})
    protocol=dict(scenes=chosen,queries_per_scene=args.queries,grid_m=.25,yaw_bins=36,ray_bins=16,ray_step_m=.1,max_range_m=20.,fp_spacing_m=.1,rgb_size=[128,96],
                  rir_samples=16384,sample_rate_hz=48000,stft_nfft=256,stft_hop=32,spectral_channels=18,spectral_cache_dtype='float16',
                  split_source_sha256=sha(args.audit/'split_manifest.json'),selection='SHA256 scene ordering then uniform schedule query subset before results',
                  test_accessed=False,evidence='Exploratory: previous development exposure remains',label_contract='continuous GT ray target for loss/diagnosis only, never inference input')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest=[]
    for part,scenes in chosen.items():
      for scene in scenes:
        start=time.perf_counter();sch_path=args.root/f'RIR/manifests/positive_pairs_v1/schedules/{part}/{scene}.json';sch=jread(sch_path)
        meta=jread(args.root/f'2DFP/{scene}/floorplan_metadata.json');free=np.array(Image.open(args.root/f'2DFP/{scene}/map.png'))==255
        spacing=meta['map_spacing_m'];origin=[meta['world_bounds_m'][k] for k in ('x_min','z_min')]
        poses,valid=candidate_bank(free,spacing,origin,.25,36)
        ranges,hits=boundary_rays(free,spacing,origin,poses[::36,:2],36)
        candidates=np.column_stack([poses,np.log1p(ranges)/math.log(21),hits]).astype(np.float32)
        selected=[sch['queries'][i] for i in np.linspace(0,len(sch['queries'])-1,min(args.queries,len(sch['queries']))).round().astype(int)]
        gt=np.array([[q['centre_xyz_m'][0],q['centre_xyz_m'][2],math.radians(q['world_yaw_deg'])] for q in selected])
        target,target_hit=rays_at_poses(free,spacing,origin,gt)
        rgbs=[];specs=[];qc=[]
        for q in selected:
            path=args.root/q['observed_rir_path'];assert '/RIR/Observed/' in str(path)
            raw=path.read_bytes();side=jread(path.with_suffix('.json'));digest=hashlib.sha256(raw).hexdigest()
            assert digest==side['rir_sha256'] and side['role']=='Observed'
            assert np.allclose(side['centre_xyz_m'],q['centre_xyz_m'])
            r=np.load(io.BytesIO(raw));assert r.shape[0]==6 and np.isfinite(r).all()
            r=canonical_rir(r,q['world_yaw_deg'],q['base_acoustic_yaw_deg'])
            audio=np.zeros((6,16384),np.float32);length=min(r.shape[-1],16384);audio[:,:length]=r[:,:length]
            rms=max(float(np.sqrt(np.mean(audio**2))),1e-8);audio/=rms
            spectrum=rir_spectrum(torch.from_numpy(audio)[None])[0].half();assert torch.isfinite(spectrum).all();specs.append(spectrum)
            rgb_path=args.root/q['image_path'];img=Image.open(rgb_path).convert('RGB').resize((128,96));rgbs.append(torch.from_numpy(np.asarray(img).copy().transpose(2,0,1)))
            qc.append(dict(query_id=q['sample_index'],rgb_path=q['image_path'],rir_path=q['observed_rir_path'],rir_sha256=digest,rgb_sha256=sha(rgb_path),raw_shape=list(r.shape),rms=rms))
        stride=round(.1/spacing);small=free[::stride,::stride]
        package=dict(scene=scene,split=part,rgb=torch.stack(rgbs),spectrum=torch.stack(specs),gt=torch.tensor(gt,dtype=torch.float32),target=torch.tensor(np.log1p(target)/math.log(21),dtype=torch.float32),
                     target_hit=torch.tensor(target_hit),candidates=torch.tensor(candidates),valid=torch.tensor(valid),floorplan=torch.tensor(np.stack([small,~small]),dtype=torch.float32)[None],
                     origin=torch.tensor(origin,dtype=torch.float32),spacing=spacing*stride,hfov_deg=106.2602,query_ids=[q['sample_index'] for q in selected])
        dest=args.out/f'{scene}.pt';torch.save(package,dest);(args.out/f'{scene}_inputs.json').write_text(json.dumps(qc,indent=2))
        row=dict(scene=scene,split=part,queries=len(selected),candidates=len(poses),cache_bytes=dest.stat().st_size,cache_sha256=sha(dest),schedule_sha256=sha(sch_path),
                 map_sha256=sha(args.root/f'2DFP/{scene}/map.png'),seconds=time.perf_counter()-start)
        manifest.append(row);(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(row),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
