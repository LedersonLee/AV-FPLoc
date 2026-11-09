"""Frozen Transfer11 fixedcandidate evaluation; allprimaryarms, no selection."""
from pathlib import Path
import json,sys
import numpy as np,torch
from oracle_capacity import represent
from candidate_conditioned import pair_score
from joint_models import JointRay
sys.path.insert(0,'/workspace/experiments/directional_structure_20261109')
from discrimination import standard,pairmask,measures,coarse
P=Path('/output');PREV=Path('/workspace/experiments/directional_structure_20261109');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text())
def load(s):
 z=dict(np.load(PREV/'candidates'/s/'pure_visual.npz'));z['v']=standard(z['vs']);z['a']=standard(z['acs']);z['geom']=np.log1p(coarse(z['geometry']));return z

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);freeze=json.loads((P/'transfer_frozen.json').read_text());assert len(freeze['models'])==18;fit=[load(s) for s in PR['training_scenes']];off=~np.eye(20,dtype=bool);cut=float(np.quantile(np.concatenate([abs(z['v'][:,:,None]-z['v'][:,None,:])[:,off].ravel() for z in fit]),1/3));highcut=float(np.quantile(np.concatenate([np.sort(z['v'],axis=1)[:,-1]-np.sort(z['v'],axis=1)[:,-2] for z in fit]),1/3));rows=[];baselines=[];rays=[]
 for r in freeze['models']:
  group,kind,seed=r['group'],r['kind'],r['seed'];root=P/'transfer_models'/f'{group}_{kind}_seed{seed}';errors=[]
  for fold,held in enumerate(PR['outer_folds']):
   orig=P/group/(f'{kind}_k36_seed{seed}_fold{fold}' if group=='acoustic' else f'{kind}_seed{seed}_fold{fold}')
   for sid in held:
    pp=np.load(orig/f'{sid}.npz')['prediction'];y=np.log1p(represent(np.load(OLD/'data'/sid/'observations.npz')['target'],'raw36'));errors.append(abs(pp-y))
  err=np.concatenate(errors);sigma=err.mean(0).clip(.02);q90=np.quantile(err,.9,axis=0);weights=1/sigma;weights/=weights.sum()
  if group=='joint' and kind=='attention':ck=torch.load(root/'model.pt',map_location='cpu',weights_only=False);model=JointRay(kind).cuda().eval();model.load_state_dict(ck['state'])
  for sid in PR['transfer_scenes']:
   z=load(sid);g=np.log1p(represent(z['geometry'],'raw36'));pred=np.load(root/f'{sid}.npz');np.testing.assert_array_equal(pred['query_ids'],z['query_ids']);p=pred['prediction'];o=np.load(OLD/'data'/sid/'observations.npz');y=represent(o['target'],'raw36');e=abs(p-np.log1p(y));rays.append(dict(scene=sid,seed=seed,group=group,kind=kind,mae=float(abs(np.clip(np.expm1(p),0,10)-y).mean()),interval90_coverage=float((e<=q90).mean()),actual_log_error=float(e.mean()),predicted_log_error=float(sigma.mean())));sv=np.sort(z['v'],axis=1);mask=pairmask(z,cut);high=sv[:,-1]-sv[:,-2]<=highcut;scores=dict(uniform=-abs(g-p[:,None]).mean(-1));controls={'matched':p}
   if (group=='acoustic' and kind in ['raw','chirp','embedding']) or (group=='joint' and kind=='attention'):
    donor=np.roll(p,17,axis=0)
    if group=='joint':
     data=np.load(P/'audio'/sid/'features.npz')['raw'];a=torch.tensor(np.clip((data-ck['mu'])/ck['std'],-10,10),dtype=torch.float32,device='cuda');v=torch.tensor(np.clip((o['visual']-ck['vm'])/ck['vs'],-10,10),dtype=torch.float32,device='cuda')
     with torch.inference_mode():donor=model(torch.roll(a,17,0),v).cpu().numpy()
    controls.update(donor17=donor,shuffled=np.roll(p,1,axis=1))
   for control,pp in controls.items():
    scs=dict(uniform=-abs(g-pp[:,None]).mean(-1))
    if len(controls)>1:scs.update(uncertainty=-(abs(g-pp[:,None])*weights).sum(-1),pair_selection=pair_score(g,pp))
    for method,score in scs.items():
     for fusion in ['geometry','visual_geometry']:
      score2=score if fusion=='geometry' else .85*z['v']+.15*standard(score);rows.append(dict(scene=sid,seed=seed,group=group,kind=kind,method=method,control=control,fusion=fusion,**measures(z,score2,mask,high,z['v'].argmax(1))))
   if group=='joint' and kind=='vision':
    aa=np.load(P/'transfer_models'/f'joint_acoustic_seed{seed}'/f'{sid}.npz')['prediction'];lp=.5*(p+aa);score=-abs(g-lp[:,None]).mean(-1)
    for fusion in ['geometry','visual_geometry']:rows.append(dict(scene=sid,seed=seed,group='joint',kind='late',method='uniform',control='matched',fusion=fusion,**measures(z,score if fusion=='geometry' else .85*z['v']+.15*standard(score),mask,high,z['v'].argmax(1))))
   if group=='acoustic' and kind=='ecn':
    for name,s in [('visual',z['v']),('cosine',z['a']),('visual_cosine',.85*z['v']+.15*z['a'])]:baselines.append(dict(scene=sid,seed=seed,arm=name,**measures(z,s,mask,high,z['v'].argmax(1))))
    b=np.load(OLD/'data'/sid/f'candidates_seed{seed}.npz')['baseline_labels']
    for n,name in [(3,'B2'),(5,'GEOQ')]:baselines.append(dict(scene=sid,seed=seed,arm=name,joint=float(b[:,n,1].mean()),fine=float(b[:,n,2].mean()),R1=float(b[:,n,0].mean())))
  if group=='joint' and kind=='attention':del model
  print('TRANSFER EVAL',group,kind,seed,flush=True)
 (P/'transfer_evaluation.json').write_text(json.dumps(dict(records=rows,ray_records=rays,baselines=baselines,pair_threshold=cut,high_threshold=highcut,scope='Frozen18models+late;reusedteacherexposedTransfer11,fixedpureVonly;scalesfromNew12OOF,GTnotusedforselection',test_accessed=False),indent=2))
if __name__=='__main__':main()
