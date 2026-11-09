"""Evaluate every completed matched model on the whole declared occlusion curve."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features
from track_b.scripts.train_foundation_interaction import read,sha
from track_b.eval.metrics import pose_statistics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--foundation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();protocol=read(a.run/'protocol.json');assert len(read(a.run/'summary.json'))==9,'Await all matched fits';a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 levels=[0,25,50,75];setup=dict(levels=levels,checkpoint_hashes={f'{m}_{s}':sha(a.run/f'{m}_{s}.pt') for m in protocol['modes'] for s in protocol['seeds']},method='frozen completed checkpoint, cached learned dual-tower maps, full SE2 bank; no changed fusion or model parameters',test_accessed=False,evidence='Exploratory reused development',source_protocol_sha256=sha(a.run/'protocol.json'));(a.out/'protocol.json').write_text(json.dumps(setup,indent=2));rows=[];mismatches=[];cost=[]
 for mode in protocol['modes']:
  for seed in protocol['seeds']:
   model=FoundationInteractionMatcher(mode).cuda().eval();model.load_state_dict(torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True));reference={(r['scene'],r['query']):r for r in read(a.run/f'{mode}_{seed}.json')['val']['rows'] if r['arm']=='normal'}
   with torch.no_grad():
    for sid in protocol['data']['scenes']['val']:
     s=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));cal={'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']};cached=model.prepare_map(s['floorplan'],cal,s['candidates'])
     for level in levels:
      folder=a.foundation/(f'features_occlusion{level}_v1' if level else 'features_v1');path=folder/f'{sid}.pt';manifest={x['scene']:x['sha256'] for x in read(folder/'manifest.json')};assert sha(path)==manifest[sid];attach_visual_features(s,torch.load(path,weights_only=True))
      for ids in torch.arange(len(s['gt']),device='cuda').split(8):
       torch.cuda.synchronize();start=time.perf_counter();field=model.forward_cached(s['visual_features'][ids].float(),s['spectrum'][ids].float(),cached);torch.cuda.synchronize();cost.append(dict(mode=mode,seed=seed,scene=sid,occlusion_percent=level,batch_queries=len(ids),cached_score_seconds=time.perf_counter()-start))
       for j,q in enumerate(ids.tolist()):
        row=dict(mode=mode,seed=seed,scene=sid,query=s['query_ids'][q],occlusion_percent=level,**pose_statistics(field[j],s['candidates'][:,:3],s['gt'][q],diverse=True));rows.append(row)
        if level==0:
         old=reference[sid,row['query']];delta={k:dict(cached=row[k],reference=old[k]) for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(row[k],old[k],atol=2e-5,rtol=1e-5)}
         if delta:mismatches.append(dict(mode=mode,seed=seed,scene=sid,query=row['query'],differences=delta))
     print('DONE',mode,seed,sid,flush=True)
   del model
 summary=[dict(mode=mode,occlusion_percent=level,**{k:float(np.mean([r[k] for r in rows if r['mode']==mode and r['occlusion_percent']==level])) for k in ['joint','r1','fine','raw20','diverse20','nll','entropy']}) for mode in protocol['modes'] for level in levels]
 for name,value in [('predictions',rows),('summary',summary),('normal_replay_mismatches',mismatches),('cached_scoring_cost',cost)]:(a.out/f'{name}.json').write_text(json.dumps(value,indent=2))
 print(json.dumps(summary));print('COMPLETE',flush=True)
if __name__=='__main__':main()
