"""Evaluate completed predictions on fixedpureV candidates; GT only for retrospective metrics."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import rankdata
from oracle_capacity import represent
sys.path.insert(0,'/workspace/experiments/directional_structure_20261109')
from discrimination import measures,standard
P=Path('/output');PREV=Path('/workspace/experiments/directional_structure_20261109');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text())
def subset(pp,k):
 t=np.arange(k)*36/k;i=np.floor(t).astype(int);w=t-i;return pp[:,i]*(1-w)+pp[:,(i+1)%36]*w

def main():
 rows=[];rayrows=[]
 for group in ['acoustic','joint']:
  for receipt in sorted((P/group).glob('*/receipt.json')):
   r=json.loads(receipt.read_text());k=r.get('k',36);seed=r['seed'];fold=r['fold'];kind=r['kind']
   for sid in r['held_scenes']:
    o=np.load(OLD/'data'/sid/'observations.npz');z=dict(np.load(PREV/'candidates'/sid/'pure_visual.npz'));mask=np.load(PREV/'discrimination'/f'pure_visual_seed{seed}_fold{fold}'/f'{sid}.npz');z['v']=mask['visual'];pred=np.load(receipt.parent/f'{sid}.npz');np.testing.assert_array_equal(pred['query_ids'],z['query_ids']);variants={'matched':pred['prediction'],'donor17':np.roll(pred['prediction'],17,axis=0),'direction_shuffle':np.roll(pred['prediction'],1,axis=1)}
    for c in ['permuted','isotropic','sorted','donor']:
     if c in pred:variants[c]=pred[c]
    for kk in ([k,4,8,12] if k==36 else [k]):
     for control,base in variants.items():
      pp=base if kk==k else subset(base,kk);target=represent(o['target'],f'raw{kk}');d=np.clip(np.expm1(pp),0,10);err=abs(d-target);ra=rankdata(pp,axis=1);rb=rankdata(target,axis=1);ra-=ra.mean(1,keepdims=True);rb-=rb.mean(1,keepdims=True);den=np.sqrt((ra*ra).sum(1)*(rb*rb).sum(1));corr=np.divide((ra*rb).sum(1),den,out=np.zeros(len(pp)),where=den>1e-12);rayrows.append(dict(group=group,kind=kind,trained_k=k,eval_k=kk,control=control,scene=sid,seed=seed,fold=fold,mae=float(err.mean()),median=float(np.median(err)),p90=float(np.quantile(err,.9)),rank=float(np.mean(corr)),direction_mae=err.mean(0).tolist()))
      g=np.log1p(represent(z['geometry'],f'raw{kk}'));score=-abs(g-pp[:,None]).mean(-1)
      for method,sc in [('geometry',score),('visual_geometry',.85*z['v']+.15*standard(score))]:rows.append(dict(group=group,kind=kind,trained_k=k,eval_k=kk,control=control,scene=sid,seed=seed,fold=fold,method=method,**measures(z,sc,mask['ambiguous_mask'],mask['high_ambiguity'],z['v'].argmax(1))))
 (P/'candidate_evaluation.json').write_text(json.dumps(dict(records=rows,ray_records=rayrows,scope='Completed receipts only; checkexpectedfitcounts beforefinal aggregation. FixedpureV proposals and fit-only thresholds reused. 36subset8 interpolates predictedlograys; native8target interpolatesmetricrays beforelog, declared difference.',test_accessed=False),indent=2));print('EVALUATED',len(rayrows),'ray records',len(rows),'candidate records')
if __name__=='__main__':main()
