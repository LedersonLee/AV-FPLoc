"""Replay a saved interaction initialization against independent fixed fusion."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features,read,sha
from track_b.eval.metrics import pose_statistics

def main():
 p=argparse.ArgumentParser()
 for k in ['initial','cache','features','reference','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',default='attention');a=p.parse_args();torch.set_num_threads(2)
 refp=read(a.reference/'protocol.json');assert refp['cache_manifest_sha256']==sha(a.cache/'manifest.json')
 refs={(r['scene'],r['query']):r for r in read(a.reference/'predictions.json') if r['seed']==a.seed and r['method']=='fixed'}
 model=FoundationInteractionMatcher(a.mode).cuda().eval();model.load_state_dict(torch.load(a.initial,weights_only=True));manifest={r['scene']:r['sha256'] for r in read(a.features/'manifest.json')};checked=set();bad=[]
 with torch.no_grad():
  for sid in refp['scene_ids']:
   scene=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));path=a.features/f'{sid}.pt';assert sha(path)==manifest[sid];attach_visual_features(scene,torch.load(path,weights_only=True))
   cache=model.prepare_map(scene['floorplan'],{'origin':scene['origin'],'spacing':scene['spacing'],'hfov_deg':scene['hfov_deg']},scene['candidates'])
   for ids in torch.arange(len(scene['query_ids']),device='cuda').split(8):
    fields=model.forward_cached(scene['visual_features'][ids].float(),scene['spectrum'][ids].float(),cache)
    for j,q in enumerate(ids.tolist()):
     key=(sid,scene['query_ids'][q]);assert key not in checked;checked.add(key);rec=pose_statistics(fields[j],scene['candidates'][:,:3],scene['gt'][q],diverse=False)
     diff={k:[rec[k],refs[key][k]] for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(rec[k],refs[key][k],rtol=1e-5,atol=2e-5)}
     if diff:bad.append(dict(scene=sid,query=key[1],differences=diff))
 assert checked==set(refs)
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'validation.json').write_text(json.dumps(dict(seed=a.seed,mode=a.mode,queries=len(checked),mismatches=bad,initial_sha256=sha(a.initial),reference_protocol_sha256=sha(a.reference/'protocol.json'),scope='Full normal cohort metric replay for this seed/initialization; not bitwise score equality'),indent=2));assert not bad,'Replay mismatch preserved';print('COMPLETE',len(checked),flush=True)
if __name__=='__main__':main()
