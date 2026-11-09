"""Frozen-model factorial intervention on score scales and learned interaction."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher

from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features,read,sha
from track_b.eval.extended_metrics import extended_pose_statistics

def main():
 p=argparse.ArgumentParser()
 for name in ['run','cache','features','out']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--reference',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--code-sha',required=True);a=p.parse_args();protocol=read(a.run/'protocol.json');assert len(read(a.run/'summary.json'))==9
 a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 assert protocol['encoder_regime']=='frozen'
 conditions=['normal','initial_scales','zero_interaction','both']
 initial_rows={(r['seed'],r['scene'],r['query']):r for r in read(a.reference/'predictions.json') if r['method']=='fixed'}
 assert read(a.reference/'protocol.json')['checkpoint_hashes']==protocol['pretrained_sha256']
 setup=dict(source_sha=a.code_sha,conditions=conditions,source_protocol_sha256=sha(a.run/'protocol.json'),smoke=a.smoke,intervention='initial_scales restores visual/acoustic log scales; zero_interaction zeroes final gate and joint-head weights/bias to yield gate1/residual0; both combines these interventions',scope='post-hoc inference interventions on frozen-encoder fits, not retraining or physical-mechanism proof',test_accessed=False,evidence='Exploratory reused six development scenes')
 (a.out/'protocol.json').write_text(json.dumps(setup,indent=2));rows=[];replay=[];initial_replay=[]
 manifest={r['scene']:r['sha256'] for r in read(a.features/'manifest.json')}
 for mode in (protocol['modes'][:1] if a.smoke else protocol['modes']):
  for seed in (protocol['seeds'][:1] if a.smoke else protocol['seeds']):
   model=FoundationInteractionMatcher(mode).cuda().eval();final_state=torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True);initial_state=torch.load(a.run/f'{mode}_{seed}_initial.pt',weights_only=True)
   reference={(r['scene'],r['query']):r for r in read(a.run/f'{mode}_{seed}.json')['val']['rows'] if r['arm']=='normal'}
   with torch.no_grad():
    for sid in (protocol['data']['scenes']['val'][:1] if a.smoke else protocol['data']['scenes']['val']):
     s=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));path=a.features/f'{sid}.pt';assert sha(path)==manifest[sid];attach_visual_features(s,torch.load(path,weights_only=True))
     for condition in conditions:
      model.load_state_dict(final_state)
      if condition in ['initial_scales','both']:
       model.log_scale.copy_(initial_state['log_scale']);model.acoustic_log_scale.copy_(initial_state['acoustic_log_scale'])
      if condition in ['zero_interaction','both']:
       for layer in [model.gate[-1],model.joint_head[-1]]:layer.weight.zero_();layer.bias.zero_()
      altered=model.prepare_map(s['floorplan'],{'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']},s['candidates'])
      for ids in torch.arange(8 if a.smoke else len(s['query_ids']),device='cuda').split(8):
       fields=model.forward_cached(s['visual_features'][ids].float(),s['spectrum'][ids].float(),altered)
       for j,q in enumerate(ids.tolist()):
        rec=dict(mode=mode,seed=seed,scene=sid,query=s['query_ids'][q],condition=condition,**extended_pose_statistics(fields[j],s['candidates'][:,:3],s['gt'][q],diverse=True));rows.append(rec)
        if condition=='normal':
         old=reference[sid,rec['query']]
         mismatch={k:[rec[k],old[k]] for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(rec[k],old[k],atol=2e-5,rtol=1e-5)}
         if mismatch:replay.append(dict(mode=mode,seed=seed,scene=sid,query=rec['query'],differences=mismatch))
        if condition=='both':
         old=initial_rows[seed,sid,rec['query']]
         mismatch={k:[rec[k],old[k]] for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(rec[k],old[k],atol=2e-5,rtol=1e-5)}
         if mismatch:initial_replay.append(dict(mode=mode,seed=seed,scene=sid,query=rec['query'],differences=mismatch))
     print('DONE',mode,seed,sid,flush=True)
   del model
 summary=[];paired=[];rng=np.random.default_rng(20261109)
 for mode in (protocol['modes'][:1] if a.smoke else protocol['modes']):
  for condition in conditions:
   rr=[r for r in rows if r['mode']==mode and r['condition']==condition]
   summary.append(dict(mode=mode,condition=condition,**{k:float(np.mean([r[k] for r in rr])) for k in ['joint','r1','fine','raw20','diverse20','nll','strict_05_15','strict_02_10']}))
   if condition=='normal':continue
   for metric in ['joint','fine','nll']:
    d=np.array([np.mean([r[metric] for r in rr if r['scene']==s])-np.mean([r[metric] for r in rows if r['mode']==mode and r['condition']=='normal' and r['scene']==s]) for s in protocol['data']['scenes']['val']]);scale=1 if metric=='nll' else 100;b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*scale
    paired.append(dict(mode=mode,condition=condition,metric=metric,delta=float(d.mean()*scale),ci95=np.quantile(b,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='scene, seeds averaged'))
 for name,obj in [('predictions',rows),('summary',summary),('paired',paired),('normal_replay_mismatches',replay),('initial_replay_mismatches',initial_replay)]:
  (a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 assert not replay and not initial_replay, 'Replay mismatch; outputs preserved for diagnosis'
 print(json.dumps(summary));print('COMPLETE',flush=True)
if __name__=='__main__':main()
