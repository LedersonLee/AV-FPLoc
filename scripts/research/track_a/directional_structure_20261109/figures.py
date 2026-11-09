from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
P=Path(__file__).resolve().parent;W=P.parents[1];OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());A=json.loads((P/'analysis.json').read_text());F=P/'figures';F.mkdir(exist_ok=True);CARD=[0,27,18,9];names=['Front (0°)','Left (+90°)','Back (180°)','Right (−90°)'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def coarse(g):return np.stack([np.median(g[...,[(c+d)%36 for d in [-2,-1,0,1,2]]],-1) for c in CARD],-1)
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
GT=[];PP=[];scene_rows=[]
for fold,held in enumerate(PR['outer_folds']):
 for sid in held:
  gt=coarse(np.load(OLD/'data'/sid/'observations.npz')['target']);p=np.clip(np.expm1(np.load(P/'prediction'/f'seed42_fold{fold}'/f'legacy_ridge_{sid}.npz')['matched']),0,10);GT.append(gt);PP.append(p);scene_rows.append((sid,gt,p))
gt=np.concatenate(GT);pp=np.concatenate(PP);fig,axs=plt.subplots(2,2,figsize=(10,9),layout='constrained')
for j,ax in enumerate(axs.flat):
 ax.scatter(gt[:,j],pp[:,j],s=8,alpha=.16,color='#31688e');ax.plot([0,10],[0,10],'--',c='gray');ax.set(xlim=(0,10),ylim=(0,10),xlabel='FP coarse extent GT (m)',ylabel='Predicted extent (m)',title=f'{names[j]}: MAE {abs(gt[:,j]-pp[:,j]).mean():.2f} m')
fig.suptitle('Acoustic-only combined ridge; outer scene-held predictions\nFP footprint supervision, not acoustic reflector ground truth');save(fig,'direction_gt_prediction')
fig,axs=plt.subplots(1,4,figsize=(14,4),layout='constrained')
for j,ax in enumerate(axs):
 t=(gt[:,j]>=3).astype(int);p=(pp[:,j]>=3).astype(int);cm=np.zeros((2,2),int);np.add.at(cm,(t,p),1);norm=cm/cm.sum(1,keepdims=True);ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
 for a in range(2):
  for b in range(2):ax.text(b,a,f'{cm[a,b]}\n{norm[a,b]:.1%}',ha='center',va='center',color='white' if norm[a,b]>.6 else 'black')
 ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Near','Open'],yticklabels=['Near','Open'],xlabel='Prediction',ylabel='FP label',title=names[j])
fig.suptitle('Near <3 m; Open ≥3 m — 1,536 unique queries, no seed duplication');save(fig,'openness_confusion')
r=json.loads((P/'prediction_results.json').read_text())['records'];fig,ax=plt.subplots(figsize=(14,5),layout='constrained');x=np.arange(12)
for off,model,color in [(-.25,'legacy_ridge','#31688e'),(0,'raw_multi','#35b779'),(.25,'raw_single','#e69f00')]:
 vals=[np.mean([t['balanced_accuracy'] for t in r if t['scene']==s and t['model']==model and t['condition']=='matched'])*100 for s in PR['training_scenes']];ax.bar(x+off,vals,.24,label=model,color=color)
ax.axhline(50,ls='--',c='gray',label='Constant prediction balanced accuracy');ax.set(xticks=x,xticklabels=PR['training_scenes'],ylim=(35,95),ylabel='Balanced near/far accuracy (%)',title='Scene-held directional prediction: same queries and folds');ax.tick_params(axis='x',rotation=40);ax.legend(fontsize=9,ncol=2);save(fig,'scene_prediction')
# Deterministic first qualifying repair and harm; never use case selection to fit a model.
cases=[]
for kind in ['repair','harm']:
 found=False
 for fold,held in enumerate(PR['outer_folds']):
  for sid in sorted(held):
   z=np.load(P/'candidates'/sid/'pure_visual.npz');s=np.load(P/'discrimination'/f'pure_visual_seed42_fold{fold}'/f'{sid}.npz');v=s['visual'].argmax(1);f=s['legacy_ridge_visual_geometry'].argmax(1);good=z['labels'][...,1];yv=good[np.arange(128),v];yf=good[np.arange(128),f];use=(~yv&yf) if kind=='repair' else (yv&~yf);use&=s['ambiguous_mask'][np.arange(128),v,f]
   if not use.any():continue
   i=int(np.flatnonzero(use)[0]);o=np.load(OLD/'data'/sid/'observations.npz');p=np.clip(np.expm1(np.load(P/'prediction'/f'seed42_fold{fold}'/f'legacy_ridge_{sid}.npz')['matched'][i]),0,10);meta=json.loads((Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD/2DFP')/sid/'floorplan_metadata.json').read_text());grid=np.array(Image.open(Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD/2DFP')/sid/'map.png'));b=meta['world_bounds_m'];sp=meta['map_spacing_m'];extent=[b['x_min'],b['x_min']+grid.shape[1]*sp,b['z_min'],b['z_min']+grid.shape[0]*sp];fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained');ax=axs[0];ax.imshow(grid,cmap='gray',origin='lower',extent=extent)
   for pose,label,color,marker in [(o['truth'][i],'GT query','#2ca02c','*'),(z['pred'][i,v[i]],'Visual choice','#d62728','x'),(z['pred'][i,f[i]],'V+structure choice','#1f77b4','o')]:
    ax.scatter(pose[0],pose[1],c=color,marker=marker,s=90,label=label);ang=np.deg2rad(pose[2]);ax.arrow(pose[0],pose[1],.8*np.cos(ang),.8*np.sin(ang),color=color,width=.04,length_includes_head=True)
   ax.set(xlabel='World X (m)',ylabel='World Z (m)',title=f'{kind}: scene {sid}, query {int(z["query_ids"][i])}');ax.legend(fontsize=8);gg=coarse(z['geometry'][i]);xx=np.arange(4);axs[1].bar(xx-.25,gg[v[i]],.25,label='Visual candidate',color='#d62728');axs[1].bar(xx,gg[f[i]],.25,label='Structural rerank candidate',color='#1f77b4');axs[1].bar(xx+.25,p,.25,label='Acoustic prediction',color='#e69f00');axs[1].set(xticks=xx,xticklabels=['Front','Left','Back','Right'],ylabel='Coarse extent (m)',title='Hypothesis geometry versus acoustic prediction');axs[1].legend(fontsize=8);save(fig,'ambiguous_pair_'+kind);cases.append(dict(kind=kind,scene=sid,fold=fold,query_index=i,query_id=int(z['query_ids'][i]),visual_slot=int(v[i]),structural_slot=int(f[i]),visual_success=bool(yv[i]),structural_success=bool(yf[i]),visual_score_gap=float(abs(s['visual'][i,v[i]]-s['visual'][i,f[i]]))));found=True;break
  if found:break
assert len(cases)==2
case=cases[0];sid=case['scene'];i=case['query_index'];fold=case['fold'];b=np.load(P/'candidates'/sid/'bank_geometry.npz');xy=b['xy'];depth=b['depth'];sm=np.stack([np.median(depth[:,[(k+d)%36 for d in [-2,-1,0,1,2]]],-1) for k in range(36)],-1);g=np.log1p(sm[:,(np.arange(36)[:,None]+np.array(CARD))%36]);pred=np.load(P/'prediction'/f'seed42_fold{fold}'/f'legacy_ridge_{sid}.npz')['matched'][i];score=-abs(g-pred).mean(-1);o=np.load(OLD/'data'/sid/'observations.npz');win=np.load(W/'experiments/hssd_scaling_20261108_1332/inputs'/sid/'windows.npy').reshape(-1,11);ang=np.deg2rad(np.arange(-5,6)*10);coord=np.tan(ang)*40*3/8+19.5;rays=np.interp(coord,np.arange(40),o['mu'][i])/np.cos(ang);sc=np.interp(coord,np.arange(40),o['scale'][i]);vs=-(abs(win-rays)/sc).sum(-1)/40;zs=lambda a:(a-a.mean())/max(a.std(),1e-8);fused=.85*zs(vs)+.15*zs(score.ravel());fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
for ax,field,title in zip(axs,[vs.reshape(-1,36),score,fused.reshape(-1,36)],['Pure visual','Acoustic structural matching','Full-bank V+structure']):
 values=field.max(1);im=ax.scatter(xy[:,0],xy[:,1],c=values,s=7,cmap='viridis');ax.scatter(*o['truth'][i,:2],c='red',marker='*',s=100,label='GT diagnostic');idx=field.argmax();ax.scatter(*xy[idx//36],c='white',edgecolors='black',marker='o',s=60,label='Selected pose');ax.set(title=title,xlabel='World X (m)',ylabel='World Z (m)',aspect='equal');fig.colorbar(im,ax=ax,shrink=.6);ax.legend(fontsize=7)
fig.suptitle(f'Score fields (max over yaw), scene {sid}, query {case["query_id"]}\nEach color scale is method-specific evidence, not a calibrated posterior');save(fig,'structural_score_field')
fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
panels=[('Pure-visual fixed Top20','discrimination',['pure_visual/visual','pure_visual/legacy_ridge_visual_geometry','pure_visual/visual_cosine'],['Visual','V+structure','V+cosine']),('Full pose bank','fullbank',['visual','visual_structure','visual_cosine'],['Visual','V+structure','V+cosine']),('B2 fixed Top20','integration',['base','matched','fixed_addition'],['Refit base','Structure refit','Frozen GEOQ\n+fixed addition'])]
for ax,(title,group,keys,labels) in zip(axs,panels):
 x=np.arange(3)
 for offset,m,color in [(-.18,'joint','#31688e'),(.18,'fine','#e69f00')]:
  vals=np.array([A[group][k][m]['mean'] for k in keys])*100;cis=np.array([A[group][k][m]['ci95'] for k in keys])*100;ax.bar(x+offset,vals,.34,label=m,color=color);ax.errorbar(x+offset,vals,yerr=np.array([vals-cis[:,0],cis[:,1]-vals]),fmt='none',c='black',capsize=3)
 ax.set(xticks=x,xticklabels=labels,ylabel='Success (%)',title=title,ylim=(0,90));ax.legend(fontsize=8)
axs[2].axhline(A['discrimination']['B2/GEOQ']['joint']['mean']*100,ls='--',c='gray',label='Original GEOQ');fig.suptitle('Scene bootstrap intervals; exploratory teacher-exposed cohorts\nCoarse acoustic structure helps fixed visual modes, not the strongest reference');save(fig,'localization_ablation');(P/'case_examples.json').write_text(json.dumps(cases,indent=2));print('FIGURES',len(list(F.glob('*.png'))))
