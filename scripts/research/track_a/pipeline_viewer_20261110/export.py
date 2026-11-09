"""Frozen inference only. Public, quantized diagnostic display assets; no training/rendering.
Run in avfploc:cu124-clean with /workspace read-only and /output writable.
"""
import argparse,gzip,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
import torch,joblib
from PIL import Image
from scipy.spatial import cKDTree
W=Path('/workspace'); E=W/'experiments'; H=E/'hssd_scaling_20261108_1332'; P=Path('/output'); D=P/'public'
R=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD')
sys.path[:0]=[str(E/'fusion_vs_posterior_20261107_1300'),str(W/'AV-FPLoc/src')]
import study as st
st.OUT=P
torch.cuda.set_per_process_memory_fraction(.20)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
modes=module('viewer_modes',E/'hypothesis_discrimination_20261107_1538/modes.py')
geo=module('viewer_geo',E/'structural_evidence_20261107_1730/inference.py')
def jread(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def save(p,x):Path(p).write_text(json.dumps(x,default=plain,allow_nan=False,indent=2)+'\n')
def gz(p,b):
 with open(p,'wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(b)
def jgz(p,x):gz(p,json.dumps(x,default=plain,allow_nan=False,separators=(',',':')).encode())
def tags(z,i):
 b=z['baseline_labels'][i,:,1];covered=z['labels'][i,:,1].any()
 return [n for n,yes in [('visual-repaired',not b[0] and (b[1] or b[2])),('visual-harmed',b[0] and not b[2]),('geoq-selection-failure',covered and not b[5]),('missing-top20',not covered),('both-ambiguous',bool((z['ambiguity'][i]>.95).all()))] if yes]
def selection(ids):
 rng=np.random.default_rng(20261110); rows=[];chosen={}
 for sid in ids:
  with np.load(H/f'candidates_seed42/{sid}.npz') as z:
   for i in range(len(z['truth'])):
    err=float(np.linalg.norm(z['baseline_pred'][i,5,:2]-z['truth'][i,:2]));rows.append(dict(scene=sid,index=i,tags=tags(z,i),error=err))
   for i in [0,*rng.choice(np.arange(1,len(z['truth'])),3,replace=False).tolist()]:chosen[(sid,i)]=['saved-field replay anchor' if i==0 else 'uniform random (seed 20261110)']
 for tag in ['visual-repaired','visual-harmed','geoq-selection-failure','missing-top20','both-ambiguous']:
  rr=sorted([r for r in rows if tag in r['tags']],key=lambda r:(r['error'],r['scene'],r['index']))
  if rr:
   for r,reason in [(rr[int(rng.integers(len(rr)))],f'{tag}: random'),(rr[len(rr)//2],f'{tag}: median GEOQ position error')]:chosen.setdefault((r['scene'],r['index']),[]).append(reason)
 save(P/'selection.json',dict(seed=20261110,source_queries=len(rows),group_counts={t:sum(t in r['tags'] for r in rows) for t in ['visual-repaired','visual-harmed','geoq-selection-failure','missing-top20','both-ambiguous']},selected=[dict(scene=s,index=i,reasons=r) for (s,i),r in sorted(chosen.items())]))
 return chosen

def cached_geometry(sid,xy):
 # Same saved graph/neighborhoods as modes.geometry; decompress each NPZ array once.
 # The historical reader decompresses inside a per-cell loop, quadratic I/O.
 with np.load(H/f'geometry/{sid}.npz') as f:z={k:f[k] for k in f.files}
 np.testing.assert_array_equal(z['xy'],xy)
 mapdir=R/'2DFP'/sid;meta=jread(mapdir/'floorplan_metadata.json');pixels=np.asarray(Image.open(mapdir/'map.png').convert('L'))==255
 rc=np.rint((xy-np.array([meta['world_bounds_m']['x_min'],meta['world_bounds_m']['z_min']]))/meta['map_spacing_m']).astype(int)
 assert ((rc[:,0]>=0)&(rc[:,0]<pixels.shape[1])&(rc[:,1]>=0)&(rc[:,1]<pixels.shape[0])).all();assert pixels[rc[:,1],rc[:,0]].all()
 tree=cKDTree(xy);eu04=tree.query_ball_tree(tree,.4+1e-7);eu1=tree.query_ball_tree(tree,1+1e-7)
 unpack=lambda key:[z['g'+key][z['p'+key][i]:z['p'+key][i+1]].tolist() for i in range(len(xy))]
 g04=unpack('04');g1=unpack('1')
 return dict(eu04=eu04,eu1=eu1,g04=g04,g1=g1,ke=modes.kernel(eu04,len(xy)),kg=modes.kernel(g04,len(xy)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args();P.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
 allids=jread(E/'ray_joint_20261109/PROTOCOL.json')['training_scenes']; split=jread(R/'manifests/scene_splits_v1.json');assert not set(allids)&set(split['test']);assert not set(allids)&set(st.CFG['selection']['train'])
 chosen=selection(allids);ids=allids[:3] if args.pilot else allids
 models={s:st.model(s,None) for s in [42,43]};rankers={s:joblib.load(E/f'structural_evidence_20261107_1730/GEOQ_seed{s}/model.joblib') for s in models}
 manifest=dict(schema=1,date='2026-11-10',split='HSSD New12 reused development / original train scenes; visual teacher exposed; not independent Test',seeds=[42,43],scenes=[],queries=[],sources=dict(study='hssd_scaling_20261108_1332',b2='mbt_remaining_suite_20261106/cap020/full_epoch8',support='hypothesis_discrimination_20261107_1538',geoq='structural_evidence_20261107_1730',export_sha256=st.sha(__file__),checkpoint_sha256={s:st.sha(models[s][2]) for s in models},external_source_sha256={str(f.relative_to(E)):st.sha(f) for f in [E/'hssd_scaling_20261108_1332/build_candidates.py',E/'fusion_vs_posterior_20261107_1300/study.py',E/'hypothesis_discrimination_20261107_1538/modes.py',E/'structural_evidence_20261107_1730/inference.py',E/'mbt_remaining_suite_20261106/model.py']}),sampling=jread(P/'selection.json'),score_units='Uncalibrated score, NOT probability or likelihood',quantization='Full XY x 36 yaw display fields; uint16 linear encoding, exact candidates stored separately. Each field records maximum reconstruction error.',no_exchange='Unavailable: no matched trained no-exchange full field exported; no feature-exchange coverage claim.',license='HSSD derived thumbnails/maps: CC BY-NC 4.0, https://3dlg-hcvc.github.io/hssd/; resized, metadata stripped; noncommercial research only.')
 checks=[]
 for sid in ids:
  folder=H/'inputs'/sid; rec=jread(folder/'receipt.json')
  for name,h in rec['files'].items():assert st.sha(folder/name)==h,(sid,name)
  q=dict(np.load(folder/'query.npz'));xy=np.load(folder/'xy.npy');win=np.load(folder/'windows.npy').astype(float);fp=torch.tensor(np.load(folder/'fp.npy'),dtype=torch.float64,device='cuda');n=len(xy)*36
  g=cached_geometry(sid,xy)
  target=D/sid;target.mkdir(exist_ok=True);meta=jread(R/f'2DFP/{sid}/floorplan_metadata.json');im=Image.open(R/f'2DFP/{sid}/map.png').convert('L');original_size=im.size;im.thumbnail((1000,1000));im.save(target/'map.png')
  jgz(target/'bank.json.gz',dict(xy=xy,world_bounds_m=meta['world_bounds_m'],map_spacing_m=meta['map_spacing_m'],original_size=original_size,image_size=im.size,yaws=list(range(0,360,10))))
  schedule=jread(R/f'RIR/manifests/positive_pairs_v1/schedules/train/{sid}.json');sched={int(r['sample_index']):r for r in schedule['queries']}
  cache={s:dict(np.load(H/f'candidates_seed{s}/{sid}.npz')) for s in models}; adapted={}; gains={}
  for seed,(model,decoder,cp) in models.items():
   z=cache[seed];assert st.sha(H/f'candidates_seed{seed}/{sid}.npz')==jread(H/f'candidates_seed{seed}/{sid}.json')['sha256'];np.testing.assert_array_equal(z['query_ids'],q['sample_indices'])
   chunks=[]
   with torch.inference_mode():
    for first in range(0,len(q['mu']),32):
     aa=[torch.tensor(q[k][first:first+32],device='cuda') for k in ['visual','audio','mu','scale','observed']];dv,da=model(aa[0],aa[1]);chunks.append([a.cpu().numpy() for a in decoder(*aa,dv,da)])
   adapted[seed]=[np.concatenate([c[k] for c in chunks]) for k in range(3)]
   ch,gain=geo.rank(rankers[seed],z['S'],z['V'][...,:11],0 if seed==42 else .01,z['V'][...,11:22]);np.testing.assert_allclose(gain,z['geoq_gain'],atol=1e-10,rtol=0);np.testing.assert_array_equal(z['indices'][np.arange(len(ch)),ch],z['baseline_indices'][:,5]);gains[seed]=(ch,gain)
  manifest['scenes'].append(dict(id=sid,xy_cells=len(xy),poses=n,input_receipt_sha256=st.sha(folder/'receipt.json')))
  selected=sorted(i for s,i in chosen if s==sid)
  for i in selected:
   sample=int(q['sample_indices'][i]);qid=f'{sid}/{sample}';fields={};raw={};seeddata={};truth=q['truth'][i]
   pr,ps=st.angular_rays(q['mu'][i].astype(float),q['scale'][i].astype(float));raw['visual']=-(abs(win-pr)/ps).sum(-1)/40
   with torch.inference_mode():raw['acoustic']=(torch.tensor(q['observed'][i:i+1],dtype=torch.float64,device='cuda')@fp.T).cpu().numpy()[0]
   fields['visual']=st.standardized(raw['visual']);fields['acoustic']=st.standardized(raw['acoustic']);fields['fusion']=.85*fields['visual']+.15*fields['acoustic']
   anchor=E/f'directional_structure_20261109/candidates/{sid}/example_fields.npz';savedcheck=None
   if i==0 and anchor.exists():
    old=np.load(anchor);print('ANCHOR KEYS',old.files,flush=True) if not checks else None
    # Saved arrays are float32, replay is float64. Compare after inspecting schema below.
    slot=int(np.flatnonzero(old['query_indices']==i)[0]);savedcheck={}
    for key,srcname in [('visual','visual'),('acoustic','cosine')]:
     err=float(abs(raw[key]-old[srcname][slot]).max());np.testing.assert_allclose(raw[key],old[srcname][slot],atol=2e-5,rtol=2e-6);savedcheck[key]=err
   for seed in models:
    mu,scale,ob=adapted[seed];r,s=st.angular_rays(mu[i].astype(float),scale[i].astype(float));v=st.standardized(-(abs(win-r)/s).sum(-1)/40)
    with torch.inference_mode():a=st.standardized((torch.tensor(ob[i:i+1],dtype=torch.float64,device='cuda')@fp.T).cpu().numpy()[0])
    b2=.85*v+.15*a;fields[f'b2_{seed}']=b2;o=modes.extract(v.reshape(-1,36),a.reshape(-1,36),xy,mu[i],scale[i],g)['support_nms'];z=cache[seed]
    np.testing.assert_array_equal(o['indices'],z['indices'][i]);np.testing.assert_array_equal(o['centers'],z['centers'][i]);np.testing.assert_allclose(o['features'],z['S'][i],atol=2e-5,rtol=1e-5)
    expected=z['baseline_indices'][i,:4];actual=np.array([fields[k].argmax() for k in ['visual','acoustic','fusion',f'b2_{seed}']]);np.testing.assert_array_equal(actual,expected)
    stages={k:modes.nms(fields[k].reshape(-1,36),g['eu1']).tolist() for k in ['visual','fusion',f'b2_{seed}']};stages['support']=o['indices'].tolist();ch,gain=gains[seed];order=np.argsort(-gain[i],kind='stable');stages['geoq']=z['indices'][i,order].tolist()
    union=sorted(set(sum(stages.values(),[])));cand=[]
    for ind in union:
     pose=[*xy[ind//36],(ind%36)*10];slots=np.flatnonzero(o['indices']==ind);c=dict(id=ind,pose=pose,visual_raw=float(raw['visual'][ind]),acoustic_cosine=float(raw['acoustic'][ind]),visual=float(fields['visual'][ind]),acoustic=float(fields['acoustic'][ind]),fusion=float(fields['fusion'][ind]),b2=float(b2[ind]),b2_visual=float(v[ind]),b2_acoustic=float(a[ind]),ranks={k:([j+1 for j,x in enumerate(vv) if x==ind]) for k,vv in stages.items()},position_error=float(np.linalg.norm(xy[ind//36]-truth[:2])),yaw_error=float(abs((pose[2]-truth[2]+180)%360-180)),support=[])
     for slot in slots:
      center=int(o['centers'][slot]);c['support'].append(dict(slot=int(slot),center_id=center,center_pose=[*xy[center//36],center%36*10],refined_id=ind,features=dict(zip(modes.NAMES,o['features'][slot])),geoq_gain=float(gain[i,slot]),chosen=bool(slot==ch[i])))
     cand.append(c)
    seeddata[seed]=dict(stages=stages,candidates=cand,selected=int(z['baseline_indices'][i,5]),support_selected=int(o['indices'][0]),geoq_threshold=0 if seed==42 else .01,geoq_choice=int(ch[i]),geoq_reason='Maximum predicted relative gain exceeds threshold' if ch[i]!=0 else 'Retain Support candidate 1 (gain threshold / baseline best)',tags=tags(z,i),ambiguity=z['ambiguity'][i],top1={k:int(fields[k].argmax()) for k in ['visual','acoustic','fusion',f'b2_{seed}']})
   metadata=[];chunks=[]
   dist=np.linalg.norm(xy-truth[:2],axis=1);yd=abs((np.arange(36)*10-truth[2]+180)%360-180);near=((dist[:,None]<=1)&(yd[None,:]<=30)).ravel()
   for name,f in fields.items():
    lo=float(f.min());hi=float(f.max());step=(hi-lo)/65535;packed=np.rint((f-lo)/step).astype('<u2');recon=packed.astype(float)*step+lo;error=float(abs(recon-f).max());assert error<=step/2+1e-10
    chunks.append(packed.tobytes());metadata.append(dict(name=name,offset=(len(chunks)-1)*n,count=n,min=lo,max=hi,mean=float(f.mean()),std=float(f.std()),error_bound=step/2,max_error=error,top1=int(f.argmax()),near_gt_max=float(f[near].max()) if near.any() else None,false_peak_max=float(f[~near].max()),raw_mean=float(raw[name].mean()) if name in raw else None,raw_std=float(raw[name].std()) if name in raw else None))
   gz(target/f'{sample}.bin.gz',b''.join(chunks));jgz(target/f'{sample}.json.gz',dict(scene=sid,sample=sample,truth=truth,fields=metadata,seeds=seeddata,selection=chosen[(sid,i)]))
   src=R/sched[sample]['image_path'];rgb=Image.open(src).convert('RGB');rgb.thumbnail((384,256));rgb.save(target/f'{sample}.webp',quality=76,method=6)
   manifest['queries'].append(dict(scene=sid,sample=sample,index=i,reasons=chosen[(sid,i)],tags={s:seeddata[s]['tags'] for s in models}))
   checks.append(dict(scene=sid,sample=sample,seeds=[42,43],baseline_exact=True,support20_exact=True,centers20_exact=True,geoq_exact=True,quantization_max_error=max(m['max_error'] for m in metadata),saved_field_schema=savedcheck));print('EXPORTED',qid,flush=True)
  del fp;torch.cuda.empty_cache()
 save(D/'manifest.json',manifest);save(P/'replay_validation.json',dict(queries=len(checks),scenes=len(ids),checks=checks,training=False,rendering=False,test_accessed=False));print('DONE',len(checks),len(ids),flush=True)
if __name__=='__main__':main()
