"""Reconstruct matched20cm score fields for reproducibly selected diagnostic cases."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from matching import laplace_field,diagnostic,pose_values,FOV
from evaluate_baseline_fields import sha


def read(p):return json.loads(Path(p).read_text())


def main():
 p=argparse.ArgumentParser()
 for k in ['workspace','study','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.workspace/'AV-FPLoc/src'))
 from avfploc.geometry.rays import angular_rays
 from avfploc.evaluation.baselines import standardized
 audit_path=a.workspace/'experiments/avloc_20261111/baseline_input_audit.json';audit=read(audit_path)
 banks=read(a.study/'banks_v1/manifest.json');br={r['scene']:r for r in banks['records']}
 manifest=read(a.study/'predictions/A2_seed42/manifest.json');assert manifest['complete'] and manifest['queries']==9965
 cohort=[];base={};joint={}
 for rec in manifest['records']:
  sid=rec['scene'];base.update({(sid,r['query_id']):r for r in read(a.study/f'baselines/fields_seed42/{sid}.json')})
  joint.update({(sid,r['query_id']):r for r in read(a.study/f'evaluation/A2_seed42_20cm/{sid}.json')})
 cohort=sorted(joint);assert len(cohort)==9965 and set(cohort)==set(base)
 rng=np.random.default_rng(20261111);cases=[('uniform_random',cohort[int(i)]) for i in rng.choice(len(cohort),2,replace=False)]
 for label,model_ok,visual_ok in [('A2_harm_extreme',False,True),('A2_repair_extreme',True,False)]:
  eligible=[k for k in cohort if bool(joint[k]['joint'])==model_ok and bool(base[k]['methods']['visual']['joint'])==visual_ok]
  if eligible:
   key=max(eligible,key=lambda k: abs(joint[k]['position_error_m']-base[k]['methods']['visual']['position_error_m']))
   cases.append((label,key))
 geopath=a.study/'baselines/geoq_replay_v1/seed42_queries.json';geo={(r['scene'],r['query_id']):r for r in read(geopath)}
 a.out.mkdir(parents=True,exist_ok=False);records=[]
 for ci,(selection,(sid,qid)) in enumerate(cases):
  rec=next(r for r in audit['records'] if r['scene']==sid);folder=a.workspace/rec['folder_relative'];sources={}
  for name,digest in {**rec['files'],'receipt.json':rec['receipt_sha256']}.items():
   assert sha(folder/name)==digest;sources[name]=digest
  with np.load(folder/'query.npz') as q:
   i=int(np.flatnonzero(q['sample_indices']==qid)[0]);truth=q['truth'][i].copy()
   xy=np.load(folder/'xy.npy');win=np.load(folder/'windows.npy').astype(float);fp=np.load(folder/'fp.npy').astype(float)
   mu,b=angular_rays(q['mu'][i].astype(float),q['scale'][i].astype(float));v=-(abs(win-mu)/b).sum(-1)/40;ac=fp@q['observed'][i].astype(float)
  fields={'visual':v,'acoustic':ac,'fixed':.85*standardized(v)+.15*standardized(ac)}
  E=a.workspace/'experiments';hroot=E/'hypothesis_discrimination_20261107_1538/cache_seed42'
  hp=next(hroot/s/(sid+'.npz') for s in ['val','extra'] if (hroot/s/(sid+'.npz')).exists());hr=read(hp.with_suffix('.json'));assert sha(hp)==hr['output_sha256']
  paths=[E/f'astra_fusion_v2_20261107_0225/support_seed42/{s}/{sid}_adapted.npz' for s in ['val','extra_val']]+[E/f'fusion_vs_posterior_20261107_1300/{s}_seed42/{sid}_adapted.npz' for s in ['cache','confirmation']]
  ap=next(p for p in paths if p.exists() and sha(p)==hr['adapted_sha256']);sources['adapted']=hr['adapted_sha256']
  with np.load(ap) as z:
   assert int(z['query_ids'][i])==qid
   mu,b=angular_rays(z['mu'][i].astype(float),z['scale'][i].astype(float));fields['b2']=.85*standardized(-(abs(win-mu)/b).sum(-1)/40)+.15*standardized(fp@z['observed'][i].astype(float))
  bp=a.study/f'banks_v1/{sid}/grid_20cm.npz';assert sha(bp)==br[sid]['files']['grid_20cm.npz'];bank=np.load(bp);np.testing.assert_allclose(xy,bank['xy'],rtol=0,atol=1e-7);sources['ray_bank']=sha(bp)
  expected={k:r for k,r in base[(sid,qid)]['methods'].items()}
  for arm,head in [('A0','V'),('A1','A'),('A2','AV')]:
   root=a.study/f'predictions/{arm}_seed42';pm=read(root/'manifest.json');pr=next(r for r in pm['records'] if r['scene']==sid);pp=root/pr['file'];assert sha(pp)==pr['sha256'];sources[arm]=pr['sha256']
   with np.load(pp) as z:
    j=int(np.flatnonzero(z['query_ids']==qid)[0]);fields[arm]=laplace_field(bank['world_rays'],z[head+'_mu'][j],z[head+'_scale'][j],FOV if head=='V' else None)
   expected[arm]=next(r for r in read(a.study/f'evaluation/{arm}_seed42_20cm/{sid}.json') if r['query_id']==qid)
  diagnostics={}
  for method,field in fields.items():
   field=np.asarray(field).reshape(-1);fields[method]=field;d=diagnostic(field,xy,truth);saved=expected[method]
   for k in ['top1_id','joint','fine','top20_nms']:assert d[k]==saved[k],(method,k)
   for k in ['position_error_m','yaw_error_deg','entropy_normalized','nearest_grid_nll','gt_region_mass']:
    np.testing.assert_allclose(d[k],saved[k],rtol=1e-9,atol=1e-9)
   diagnostics[method]={k:d[k] for k in ['top1_id','joint','fine','position_error_m','yaw_error_deg','entropy_normalized','raw_peak_margin','distinct_peak_margin']}
  gp=pose_values(xy,[geo[(sid,qid)]['selected_id']])[0]
  for mode in ['native_logmass','standardized_score']:
   fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained');lo,hi=(-16,0) if mode=='native_logmass' else (-3,3)
   for ax,(method,field) in zip(axes.flat,fields.items()):
    values=field-logsumexp(field) if mode=='native_logmass' else standardized(field)
    values=values.reshape(-1,36).max(1);im=ax.scatter(xy[:,0],xy[:,1],c=values,s=7,marker='s',cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
    ax.scatter(*truth[:2],marker='*',s=110,c='white',edgecolors='black',label='GT (diagnostic)')
    pose=pose_values(xy,[diagnostics[method]['top1_id']])[0];ax.scatter(*pose[:2],marker='x',s=55,c='red',label='Top1')
    for pp,color in [(truth,'white'),(pose,'red')]:ax.arrow(pp[0],pp[1],.45*np.cos(np.deg2rad(pp[2])),.45*np.sin(np.deg2rad(pp[2])),color=color,width=.025,length_includes_head=True)
    if method=='b2':ax.scatter(*gp[:2],marker='D',s=35,facecolors='none',edgecolors='cyan',label='GEOQ selection')
    dd=diagnostics[method];ax.set_title(f"{method} | error {dd['position_error_m']:.2f}m / {dd['yaw_error_deg']:.0f}°\nH={dd['entropy_normalized']:.3f}; Joint={dd['joint']}",fontsize=10);ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
   axes.flat[7].axis('off');axes.flat[7].text(0,.95,'20cm XY / 10° yaw; seed42\nXY maximum over yaw (not marginal)\nWhite star/arrow: GT diagnostic\nRed cross/arrow: inferred Top1\nCyan diamond: GEOQ selected pose\nBackground: valid pose-grid cells\n\nNative: log normalized score mass, T=1\nNot demonstrated calibrated probability\nz-score: within-method full-pose scaling\nColor limits shared within each figure\n\nNo RGB/RIR/raw arrays included.\nGEOQ utility has no global score field.',va='top',fontsize=11)
   label='log normalized score mass (clipped at −16)' if mode=='native_logmass' else 'within-method z-score (clipped to ±3)'
   fig.colorbar(im,ax=list(axes.flat[:7]),label=label,shrink=.65)
   fig.suptitle(f'{selection} | scene {sid} | query {qid}\nReused Val19 diagnostic; same query and pose grid',fontsize=14)
   fig.savefig(a.out/f'case_{ci}_{mode}.png',dpi=140);plt.close(fig)
  records.append({'selection':selection,'scene':sid,'query_id':qid,'sources':sources,'diagnostics':diagnostics})
 result={'complete':True,'cases':records,'field_replays':sum(len(r['diagnostics']) for r in records),'selection':'2 uniform queries seed20261111; max positional-error difference in A2 harm/repair versus existing visual. Extremes are diagnostic, not representative frequency.','source_sha256':sha(__file__),'audit_sha256':sha(audit_path),'geoq_sha256':sha(geopath),'test_accessed':False,'scope':'Exploratory reused Val19 seed42. A3 pending. Native score scales differ; z-scores are visualization only.'}
 (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['field_replays'],'field replays;',len(records),'cases',flush=True)
if __name__=='__main__':main()
