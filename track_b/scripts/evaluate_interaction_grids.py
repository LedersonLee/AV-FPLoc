"""All completed joint models on the same frozen six-grid sensitivity protocol."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features,read,sha
from track_b.eval.extended_metrics import extended_pose_statistics,reliability_statistics

def main():
 p=argparse.ArgumentParser()
 for k in ['run','cache','grid_cache','features','out']:p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
 p.add_argument('--code-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
 training=read(a.run/'protocol.json');assert len(read(a.run/'summary.json'))==9
 gp=read(a.grid_cache/'protocol.json');assert gp['reference_protocol_sha256']==sha(a.cache/'protocol.json') and gp['scenes']==training['data']['scenes']['val']
 manifest={(r['scene'],r['variant']):r for r in read(a.grid_cache/'manifest.json')};assert len(manifest)==36
 features={r['scene']:r['sha256'] for r in read(a.features/'manifest.json')}
 assert sha(a.features/'protocol.json')==training['visual_cache_protocol_sha256']
 a.out.mkdir(parents=True,exist_ok=False);setup=dict(code_sha=a.code_sha,grid_protocol=gp,training_protocol_sha256=sha(a.run/'protocol.json'),grid_manifest_sha256=sha(a.grid_cache/'manifest.json'),checkpoint_hashes={f'{m}_{s}':sha(a.run/f'{m}_{s}.pt') for m in training['modes'] for s in training['seeds']},observation_batch=8,validity='nearest raster cell in declared bank, never clamp/drop GT; all-query and valid-subset',evidence='Exploratory reused development; all fixed conditions, no retraining/calibration/selection',test_accessed=False);(a.out/'protocol.json').write_text(json.dumps(setup,indent=2))
 rows=[];replay=[]
 for mode in training['modes']:
  for seed in training['seeds']:
   model=FoundationInteractionMatcher(mode).cuda().eval();model.load_state_dict(torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True))
   reference={(r['scene'],r['query']):r for r in read(a.run/f'{mode}_{seed}.json')['val']['rows'] if r['arm']=='normal'}
   with torch.no_grad():
    for sid in gp['scenes']:
     s=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));path=a.features/f'{sid}.pt';assert sha(path)==features[sid];attach_visual_features(s,torch.load(path,weights_only=True))
     for variant,step,yaws,erosion in gp['variants']:
      path=a.grid_cache/f'{sid}_{variant}.pt';assert sha(path)==manifest[sid,variant]['sha256'];bank=torch.load(path,weights_only=True);candidates=bank['candidates'].cuda();poses=candidates[:,:3];valid=bank['valid'].numpy()
      ij=np.rint((s['gt'][:,:2].cpu().numpy()-s['origin'].cpu().numpy())/step).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<valid.shape[1])&(ij[:,1]>=0)&(ij[:,1]<valid.shape[0]);included=np.zeros(len(ij),bool);included[inside]=valid[ij[inside,1],ij[inside,0]]
      cached=model.prepare_map(s['floorplan'],{'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']},candidates)
      for ids in torch.arange(len(s['query_ids']),device='cuda').split(8):
       field=model.forward_cached(s['visual_features'][ids].float(),s['spectrum'][ids].float(),cached)
       for j,q in enumerate(ids.tolist()):
        row=dict(mode=mode,seed=seed,scene=sid,query=s['query_ids'][q],variant=variant,gt_grid_valid=bool(included[q]),**extended_pose_statistics(field[j],poses,s['gt'][q],diverse=True));row['nll_gain_over_uniform']=math.log(len(poses))-row['nll'];rows.append(row)
        if variant=='baseline':
         old=reference[sid,row['query']];differences={k:[row[k],old[k]] for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(row[k],old[k],atol=2e-5,rtol=1e-5)}
         if differences:replay.append(dict(mode=mode,seed=seed,scene=sid,query=row['query'],differences=differences))
      print('DONE',mode,seed,sid,variant,flush=True)
   del model
 summary=[];reliability=[];paired=[];rng=np.random.default_rng(20261109)
 keys=['joint','r1','fine','raw5','raw10','raw20','diverse20','strict_05_15','strict_02_10','nll','nll_gain_over_uniform','normalized_entropy','oracle_joint','oracle_fine']
 for mode in training['modes']:
  for variant,*_ in gp['variants']:
   rr=[r for r in rows if r['mode']==mode and r['variant']==variant];vv=[r for r in rr if r['gt_grid_valid']]
   summary.append(dict(mode=mode,variant=variant,queries=len(rr),grid_invalid_rate=1-len(vv)/len(rr),all_query={k:float(np.mean([r[k] for r in rr])) for k in keys},valid_subset={k:float(np.mean([r[k] for r in vv])) for k in keys} if vv else None))
   reliability.append(dict(mode=mode,variant=variant,**reliability_statistics(rr)))
   for metric in ['joint','fine','raw20','diverse20']:
    delta=np.array([np.mean([r[metric] for r in rr if r['scene']==s])-np.mean([r[metric] for r in rows if r['mode']==mode and r['variant']=='baseline' and r['scene']==s]) for s in gp['scenes']]);b=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)*100
    paired.append(dict(mode=mode,variant=variant,metric=metric,delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(b,[.025,.975]).tolist(),unit='scene; seeds averaged'))
 for name,obj in [('predictions',rows),('summary',summary),('reliability',reliability),('paired',paired),('baseline_replay_mismatches',replay)]:(a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
