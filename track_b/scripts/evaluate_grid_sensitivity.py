"""Fixed strong F1 checkpoints on every declared exhaustive grid/mask variant."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features,full_descriptors
from track_b.scripts.train_foundation_interaction import read,sha
from track_b.eval.extended_metrics import extended_pose_statistics,reliability_statistics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--grid-cache',type=Path,required=True);ap.add_argument('--foundation',type=Path,required=True);ap.add_argument('--acoustic-run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 protocol=read(a.grid_cache/'protocol.json');assert protocol['reference_protocol_sha256']==sha(a.cache/'protocol.json');manifest={(x['scene'],x['variant']):x for x in read(a.grid_cache/'manifest.json')};assert len(manifest)==36;feature_manifest={r['scene']:r['sha256'] for r in read(a.foundation/'features_v1/manifest.json')};reference={(r['scene'],r['seed'],r['query'],r['method']):r for r in read(a.foundation/'independent_fusion_anchor_v1/predictions.json')};rows=[];cost=[];mismatches=[]
 setup=dict(observation_encoding_batch=8,grid_protocol=protocol,grid_manifest_sha256=sha(a.grid_cache/'manifest.json'),seeds=[42,43,44],weights=[.85,.15],temperature=1,checkpoint_hashes={f'{mode}_{seed}':sha(folder/f'{mode}_{seed}.pt') for mode,folder in [('visual',a.foundation/'learned_fp_w1000_anchor1000_v1'),('acoustic',a.acoustic_run)] for seed in [42,43,44]},validity='GT nearest raster cell in declared candidate validity; out-of-grid/invalid flagged, never clamped or dropped; separate all-query and valid-subset',evidence='Exploratory six reused development scenes; fixed checkpoints and no calibration refit; no Test')
 (a.out/'protocol.json').write_text(json.dumps(setup,indent=2))
 for seed in setup['seeds']:
  vm=FoundationStructureMatcher().cuda().eval();vm.load_state_dict(torch.load(a.foundation/f'learned_fp_w1000_anchor1000_v1/visual_{seed}.pt',weights_only=True));am=StructureMatcher().cuda().eval();am.load_state_dict(torch.load(a.acoustic_run/f'acoustic_{seed}.pt',weights_only=True))
  with torch.no_grad():
   for sid in protocol['scenes']:
    s=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));path=a.foundation/f'features_v1/{sid}.pt';assert sha(path)==feature_manifest[sid];attach_visual_features(s,torch.load(path,weights_only=True));v=torch.cat([vm.visual(part.float()) for part in s['visual_features'].split(8)]);au=torch.cat([am.acoustic(part.float()) for part in s['spectrum'].split(8)])
    for variant,step,yaws,erosion in protocol['variants']:
     path=a.grid_cache/f'{sid}_{variant}.pt';assert sha(path)==manifest[sid,variant]['sha256'];bank=torch.load(path,weights_only=True);s['candidates']=bank['candidates'].cuda();poses=s['candidates'][:,:3];valid=bank['valid'].numpy();ij=np.rint((s['gt'][:,:2].cpu().numpy()-s['origin'].cpu().numpy())/step).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<valid.shape[1])&(ij[:,1]>=0)&(ij[:,1]<valid.shape[0]);included=np.zeros(len(ij),bool);included[inside]=valid[ij[inside,1],ij[inside,0]]
     torch.cuda.synchronize();start=time.perf_counter();vf=full_descriptors(vm,s,True);af=full_descriptors(am,s,True);torch.cuda.synchronize();cost.append(dict(seed=seed,scene=sid,variant=variant,candidates=len(poses),map_encoding_seconds=time.perf_counter()-start));weight=.25+.75*s['candidates'][:,19:35];mask=visible_rays(s['hfov_deg'],device='cuda')
     for ids in torch.arange(len(s['gt']),device='cuda').split(8):
      sv=vm.match(v[ids],vf,weight*mask);sa=am.match(au[ids],af,weight)
      for method,field in [('visual',sv),('acoustic',sa),('fixed',.85*sv+.15*sa)]:
       for j,q in enumerate(ids.tolist()):
        row=dict(seed=seed,scene=sid,query=s['query_ids'][q],variant=variant,method=method,gt_grid_valid=bool(included[q]),**extended_pose_statistics(field[j],poses,s['gt'][q],diverse=True));row['nll_gain_over_uniform']=math.log(len(poses))-row['nll'];rows.append(row)
        if variant=='baseline':
         old=reference[sid,seed,row['query'],method];diff={k:dict(current=row[k],reference=old[k]) for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(row[k],old[k],atol=2e-5,rtol=1e-5)}
         if diff:mismatches.append(dict(scene=sid,seed=seed,query=row['query'],method=method,differences=diff))
     print('DONE',seed,sid,variant,flush=True)
 summary=[];reliability=[];keys=['joint','r1','fine','raw5','raw10','raw20','diverse20','strict_05_15','strict_02_10','nll','nll_gain_over_uniform','normalized_entropy','oracle_joint','oracle_fine']
 for variant,*_ in protocol['variants']:
  for method in ['visual','acoustic','fixed']:
   rr=[r for r in rows if r['variant']==variant and r['method']==method];vv=[r for r in rr if r['gt_grid_valid']];summary.append(dict(variant=variant,method=method,queries=len(rr),grid_invalid_rate=1-len(vv)/len(rr),all_query={k:float(np.mean([r[k] for r in rr])) for k in keys},valid_subset={k:float(np.mean([r[k] for r in vv])) for k in keys} if vv else None));reliability.append(dict(variant=variant,method=method,**reliability_statistics(rr)))
 rng=np.random.default_rng(20261109);paired=[]
 for variant,*_ in protocol['variants']:
  for metric in ['joint','fine','raw20','strict_05_15','strict_02_10']:
   delta=np.array([np.mean([r[metric] for r in rows if r['scene']==s and r['variant']==variant and r['method']=='fixed'])-np.mean([r[metric] for r in rows if r['scene']==s and r['variant']=='baseline' and r['method']=='fixed']) for s in protocol['scenes']]);boot=delta[rng.integers(0,6,(10000,6))].mean(1)*100;paired.append(dict(variant=variant,metric=metric,delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),unit='six scenes; seeds averaged'))
 for name,value in [('predictions',rows),('summary',summary),('reliability',reliability),('paired_scene_bootstrap',paired),('baseline_replay_mismatches',mismatches),('map_cost',cost)]:(a.out/f'{name}.json').write_text(json.dumps(value,indent=2))
 print(json.dumps(summary));print('COMPLETE',flush=True)
if __name__=='__main__':main()
