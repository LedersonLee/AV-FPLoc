"""Exact Phase4 fullbank scores, share GT-free geometry selection across allmodels/controls."""
from candidate_conditioned import *
def main():
 records=[]
 for fold,held in enumerate(PR['outer_folds']):
  for sid in held:
   active=[]
   for seed in PR['seeds']:
    for group,kind in [('acoustic','raw'),('acoustic','chirp'),('acoustic','embedding'),('joint','attention')]:
     dest=P/'conditioned_fullbank'/f'{group}_{kind}_seed{seed}_fold{fold}';dest.mkdir(parents=True,exist_ok=True);receipt=dest/f'{sid}.json'
     if receipt.exists():records+=json.loads(receipt.read_text())['records'];continue
     root=P/group/(f'{kind}_k36_seed{seed}_fold{fold}' if group=='acoustic' else f'{kind}_seed{seed}_fold{fold}');pp=np.load(root/f'{sid}.npz')['prediction'];sigma=np.load(P/'calibration'/f'{group}_{kind}_seed{seed}_fold{fold}'/'scales.npz')['sigma'];weights=1/sigma;weights/=weights.sum();active.append(dict(seed=seed,group=group,kind=kind,dest=dest,receipt=receipt,p=pp,weights=weights,choices={},labels={}))
   if not active:continue
   start=time.time();z=dict(np.load(PREV/'candidates'/sid/'pure_visual.npz'));o=np.load(OLD/'data'/sid/'observations.npz');bg=np.load(PREV/'candidates'/sid/'bank_geometry.npz');xy=bg['xy'];dep=bg['depth'];dense=dep[:,(np.arange(36)[:,None]+np.arange(36))%36].reshape(-1,36);bank=np.log1p(represent(dense,'raw36'));nb=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);win=np.load(H/'inputs'/sid/'windows.npy').reshape(-1,11)
   for i in range(128):
    r,s=angular_rays(o['mu'][i].astype(float),o['scale'][i].astype(float));visual=standardized((-abs(win-r)/s).sum(-1)/40);anchor=int(z['indices'][i,z['vs'][i].argmax()]);w=selection_weights(abs(bank-bank[anchor]))
    for a in active:
     for control,pp in [('matched',a['p'][i]),('donor17',a['p'][(i-17)%128]),('shuffled',np.roll(a['p'][i],1))]:
      e=abs(bank-pp);scores=dict(uniform=-e.mean(-1),uncertainty=-(e*a['weights']).sum(-1),anchor_selection=(w*(e[anchor]-e)).sum(-1))
      for method,score in scores.items():
       for fusion in ['geometry','visual_geometry']:
        sc=score if fusion=='geometry' else .85*visual+.15*standardized(score);idx=nms(sc,nb);d=np.linalg.norm(xy[idx//36]-o['truth'][i,:2],axis=-1);yaw=abs((idx%36*10-o['truth'][i,2]+180)%360-180);key=control+'_'+method+'_'+fusion;a['choices'].setdefault(key,[]).append(idx);a['labels'].setdefault(key,[]).append(np.stack([d<=1,(d<=1)&(yaw<=30),d<=.1],-1))
    if i%32==31:print('SHARED FULLBANK',sid,i+1,round(time.time()-start,1),flush=True)
   for a in active:
    rows=[];arrays={}
    for control in ['matched','donor17','shuffled']:
     for method in ['uniform','uncertainty','anchor_selection']:
      for fusion in ['geometry','visual_geometry']:
       key=control+'_'+method+'_'+fusion;lab=np.array(a['labels'][key]);rows.append(dict(scene=sid,seed=a['seed'],fold=fold,group=a['group'],kind=a['kind'],control=control,method=method,fusion=fusion,joint=float(lab[:,0,1].mean()),fine=float(lab[:,0,2].mean()),R1=float(lab[:,0,0].mean()),top5=float(lab[:,:5,1].any(1).mean()),top20=float(lab[:,:,1].any(1).mean())));arrays[key+'_indices']=np.array(a['choices'][key]);arrays[key+'_labels']=lab
    np.savez_compressed(a['dest']/f'{sid}.npz',**arrays,query_ids=o['query_ids']);a['receipt'].write_text(json.dumps(dict(records=rows,seconds_shared_scene=time.time()-start,implementation='Exactsharedgeometry; existingblocksreused'),indent=2));records+=rows
 (P/'conditioned_fullbank_results.json').write_text(json.dumps(dict(records=records,scope='SamePhase4scores; sharedFPdirectionselection acrossmodels/controlsinstead ofrecomputing.',test_accessed=False),indent=2))
if __name__=='__main__':main()
