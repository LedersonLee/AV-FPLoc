"""Nondeployable ray capacity diagnostic; no training and no Test inputs."""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/workspace/experiments/directional_structure_20261109')
from prepare_candidates import nms
from discrimination import measures,load
P=Path('/output');PREV=Path('/workspace/experiments/directional_structure_20261109');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text())
# Imported load points at /output by design; load pureV explicitly from immutable prior study.
def rays(x,angles):
 t=np.mod(-np.asarray(angles)/10,36);lo=np.floor(t).astype(int);w=t-lo;return x[...,lo]*(1-w)+x[...,(lo+1)%36]*w
def represent(x,arm):
 if arm=='smooth4':return np.stack([np.median(x[...,[(c+j)%36 for j in [-2,-1,0,1,2]]],-1) for c in [0,27,18,9]],-1)
 if arm=='subset8':return x[...,np.mod(-np.round(np.arange(8)*4.5).astype(int),36)]
 if arm=='front12':return rays(x,np.arange(-50,61,10))
 return rays(x,np.arange(int(arm[3:]))*360/int(arm[3:]))
def main():
 rows=[];dst=P/'oracle';dst.mkdir(exist_ok=True)
 for fold,held in enumerate(PR['outer_folds']):
  for sid in held:
   final=dst/(sid+'.json')
   if final.exists():rows+=json.loads(final.read_text())['records'];continue
   start=time.time();o=np.load(OLD/'data'/sid/'observations.npz');bg=np.load(PREV/'candidates'/sid/'bank_geometry.npz');xy=bg['xy'];depth=bg['depth'];nb=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);z=dict(np.load(PREV/'candidates'/sid/'pure_visual.npz'));m=np.load(PREV/'discrimination'/f'pure_visual_seed42_fold{fold}'/f'{sid}.npz');z['v']=m['visual'];base=z['v'].argmax(1);per=[];saved={}
   # Every candidate has geometry at36relative heading offsets, no change to pose bank.
   dense=depth[:,(np.arange(36)[:,None]+np.arange(36))%36].reshape(-1,36)
   for arm in ['raw4','raw8','raw12','raw36','subset8','front12','smooth4']:
    target=np.log1p(represent(o['target'],arm));bank=np.log1p(represent(dense,arm));cand=np.log1p(represent(z['geometry'],arm));score=-abs(cand-target[:,None]).mean(-1);stats=measures(z,score,m['ambiguous_mask'],m['high_ambiguity'],base);per.append(dict(scene=sid,arm=arm,scope='fixed_pureV',**stats));ids=[];labs=[]
    for i in range(len(target)):
     idx=nms(-abs(bank-target[i]).mean(-1),nb);dist=np.linalg.norm(xy[idx//36]-o['truth'][i,:2],axis=-1);yaw=abs((idx%36*10-o['truth'][i,2]+180)%360-180);lab=np.stack([dist<=1,(dist<=1)&(yaw<=30),dist<=.1],-1);ids.append(idx);labs.append(lab)
    labs=np.array(labs);per.append(dict(scene=sid,arm=arm,scope='fullbank',joint=float(labs[:,0,1].mean()),fine=float(labs[:,0,2].mean()),R1=float(labs[:,0,0].mean()),top5=float(labs[:,:5,1].any(1).mean()),top20=float(labs[:,:,1].any(1).mean())));saved[arm+'_indices']=np.array(ids);saved[arm+'_labels']=labs;saved[arm+'_candidate_scores']=score
   # Exact36 output subsetting equals lower-K oracle when directions coincide;8 interpolation distinguished.
   np.testing.assert_array_equal(represent(o['target'],'raw4'),represent(o['target'],'raw36')[:,::9]);np.testing.assert_array_equal(represent(o['target'],'raw12'),represent(o['target'],'raw36')[:,::3]);np.savez_compressed(dst/(sid+'.npz'),**saved,query_ids=o['query_ids']);final.write_text(json.dumps(dict(records=per,seconds=time.time()-start,oracle_only=True),indent=2));rows+=per;print('ORACLE',sid,round(time.time()-start,1),flush=True)
 (P/'oracle_results.json').write_text(json.dumps(dict(records=rows,scope='Nondeployable oracle; observedpose ray truth is diagnostic only. Same36 source, continuousquery vs discretecandidate mismatch retained. No acoustic training.',test_accessed=False),indent=2))
if __name__=='__main__':main()
