from pathlib import Path
import json,hashlib,time,joblib,subprocess
import numpy as np
from prediction import fit,predict,target
from discrimination import load,coarse,standard,pairmask,measures
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());dst=P/'transfer';dst.mkdir(exist_ok=True)
def main():
 ids=PR['training_scenes'];held=PR['transfer_scenes'];assert not set(ids)&set(held);obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in ids};x=np.concatenate([np.c_[obs[s]['features'],obs[s]['embedding']] for s in ids]);y=np.concatenate([target(obs[s]) for s in ids]);path=dst/'combined_ridge.joblib'
 if path.exists():m=joblib.load(path)
 else:m=fit(x,y);joblib.dump(m,path)
 ztrain=[load(s,42,'pure_visual') for s in ids];off=~np.eye(20,dtype=bool);threshold=float(np.quantile(np.concatenate([abs(z['v'][:,:,None]-z['v'][:,None,:])[:,off].ravel() for z in ztrain]),1/3));highcut=float(np.quantile(np.concatenate([np.sort(z['v'],axis=1)[:,-1]-np.sort(z['v'],axis=1)[:,-2] for z in ztrain]),1/3));frozen=dict(fit_scenes=ids,held_scenes=held,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pair_threshold=threshold,high_ambiguity_threshold=highcut,code_commit=subprocess.check_output(['git','-C',str(W/'AV-FPLoc'),'rev-parse','HEAD'],text=True).strip(),coefficient=.15,created_before_newmethod_transfer_evaluation=True);(dst/'frozen_before_evaluation.json').write_text(json.dumps(frozen,indent=2));records=[];baselines=[]
 for sid in held:
  o=dict(np.load(OLD/'data'/sid/'observations.npz'));p=predict(m,np.c_[o['features'],o['embedding']]);z=load(sid,42,'pure_visual');score=-abs(z['geom']-p[:,None]).mean(-1);prior=-abs(z['geom']-y.mean(0)).mean(-1);donor=-abs(z['geom']-np.roll(p,17,axis=0)[:,None]).mean(-1);scores=dict(visual=z['v'],cosine=z['a'],visual_cosine=.85*z['v']+.15*z['a'],structure=score,visual_structure=.85*z['v']+.15*standard(score),prior=prior,donor=donor,visual_donor=.85*z['v']+.15*standard(donor));mask=pairmask(z,threshold);sv=np.sort(z['v'],axis=1);high=sv[:,-1]-sv[:,-2]<=highcut
  for arm,g in scores.items():records.append(dict(scene=sid,arm=arm,**measures(z,g,mask,high,z['v'].argmax(1))))
  np.savez_compressed(dst/(sid+'.npz'),**scores,predicted_log_geometry=p,query_ids=z['query_ids'])
  for seed in [42,43]:
   b=np.load(OLD/'data'/sid/f'candidates_seed{seed}.npz')['baseline_labels']
   for k,name in [(2,'fullbank_fixed_fusion'),(3,'B2'),(4,'support'),(5,'GEOQ')]:baselines.append(dict(scene=sid,seed=seed,arm=name,joint=float(b[:,k,1].mean()),fine=float(b[:,k,2].mean()),R1=float(b[:,k,0].mean())))
 (P/'transfer_results.json').write_text(json.dumps(dict(records=records,baselines=baselines,frozen=frozen,scope='Exploratory teacher-exposed/research-reused Transfer11; no newmethod tuning after freeze;not independent validation.',test_accessed=False),indent=2));print('TRANSFER COMPLETE',len(held),flush=True)
if __name__=='__main__':main()
