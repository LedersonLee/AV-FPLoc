"""Post-hoc Study31 figures from saved predictions only; no model or training imports."""
import argparse,json,hashlib,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

COL={'gt':'#202d3a','normal':'#007f9b','scalar':'#e8a027','rotated':'#9052ad','donor':'#d95245','phase':'#82928b'}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-7 and np.std(b)>1e-7 else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);inputs={};rows=[];cache={};verified=0
 def load(path):
  inputs[str(path.relative_to(a.study))]=sha(path)
  with np.load(path) as z:return {k:z[k].copy() for k in z.files}
 frozen=read(a.study/'method_freeze_v1.json');scenes=frozen['official_test'];models=['echo_regression','echo_categorical','spatial_regression','spatial_categorical']
 refs={s:read(a.study/'official_targets_v1'/f'{s}_queries.json') for s in scenes}
 for s in scenes:inputs[str(Path('official_targets_v1')/f'{s}_queries.json')]=sha(a.study/'official_targets_v1'/f'{s}_queries.json')
 for model in models:
  for seed in [42,43,44]:
   for scene in scenes:
    root=a.study/'official_evaluation_v1/furnished/representation_v1'/f'{model}_{seed}'
    z=load(root/f'{scene}__normal.npz');don=load(root/f'{scene}__same_scene_different_position.npz')
    assert np.array_equal(z['target'],don['target']) and np.array_equal(z['mask'],don['mask']) and np.array_equal(z['donor'],don['donor'])
    m=z['mean'].astype(float);y=z['target'].astype(float);mask=z['mask'];d=don['mean'].astype(float);sc=np.repeat(m.mean(1,keepdims=True),36,axis=1);rot=np.roll(m,18,axis=1)
    assert m.shape==(64,36) and mask.dtype==bool and np.isfinite(m).all()
    # Saved donor is a different location, not a different yaw at the same centre.
    xy=np.array([q['centre_xyz_m'] for q in refs[scene]])[:,[0,2]];assert np.all(np.linalg.norm(xy-xy[z['donor']],axis=1)>.5)
    assert np.max(abs(d-m[z['donor']]))<.01  # Different inference batch grouping may introduce small numeric differences.
    verified+=1
    for i in range(64):
     good=mask[i];n=int(good.sum());assert n>2
     errors={k:float(np.abs(v[i]-y[i])[good].mean()) for k,v in [('normal',m),('scalar',sc),('rotated',rot),('donor',d)]}
     row=dict(model=model,seed=seed,scene=scene,index=i,query_id=refs[scene][i]['query_id'],donor_index=int(z['donor'][i]),donor_query_id=refs[scene][int(z['donor'][i])]['query_id'],valid=n,mae=errors,correlation=corr(m[i,good],y[i,good]),prediction_angular_sd=float(m[i].std()),gt_valid_angular_sd=float(y[i,good].std()),prediction_valid_mean=float(m[i,good].mean()),gt_valid_mean=float(y[i,good].mean()),prediction_change=float(np.abs(m[i]-d[i]).mean()))
     row['delta']={k:errors[k]-errors['normal'] for k in ['scalar','rotated','donor']};rows.append(row)
    if model=='spatial_regression':cache[scene,seed]=dict(normal=m,gt=y,mask=mask,scalar=sc,rotated=rot,donor=d,donor_index=z['donor'])
 primary=[r for r in rows if r['model']=='spatial_regression'];seed42=[r for r in primary if r['seed']==42];eligible=[r for r in seed42 if r['valid']>=18];chosen=[]
 def choose(label,ordered):
  r=next(r for r in ordered if (r['scene'],r['index']) not in {(c['scene'],c['index']) for c in chosen});chosen.append(dict(r,label=label))
 for label,quantile in [('Representative A (40th MAE percentile)',.4),('Representative B (60th MAE percentile)',.6)]:
  q=np.quantile([r['mae']['normal'] for r in eligible],quantile);pool=[r for r in eligible if r['scene'] not in [c['scene'] for c in chosen]];choose(label,sorted(pool,key=lambda r:abs(r['mae']['normal']-q)))
 for label in ['Failure A (largest eligible MAE)','Failure B (largest MAE in another scene)']:
  pool=[r for r in eligible if r['scene'] not in [c['scene'] for c in chosen[2:]];choose(label,sorted(pool,key=lambda r:-r['mae']['normal']))
 choose('Donor-sensitive (largest mean prediction change)',sorted(eligible,key=lambda r:-r['prediction_change']))
 median=np.median([r['mae']['normal'] for r in eligible]);pool=[r for r in eligible if abs(r['mae']['normal']-median)<.3 and r['gt_valid_angular_sd']>1]
 choose('Scalar-equivalent (smallest absolute MAE delta)',sorted(pool,key=lambda r:abs(r['delta']['scalar'])))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white','figure.facecolor':'white'})
 angles=np.arange(36)*10
 def finish(fig,name):fig.savefig(a.out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
 def panel(ax,r):
  v=cache[r['scene'],r['seed']];i=r['index'];good=v['mask'][i]
  ax.plot(angles,np.where(good,v['gt'][i],np.nan),'.-',color=COL['gt'],lw=2,label='GT (valid directions)')
  for k,ls,lw in [('normal','-',2.8),('scalar',':',2.6),('rotated','--',1.6),('donor','-.',1.8)]:ax.plot(angles,v[k][i],ls,color=COL[k],lw=lw,label={'normal':'Original','scalar':'Scalarized','rotated':'Rotated 180°','donor':'Donor RIR'}[k])
  ax.set(xlim=(0,350),ylim=(0,max(v['gt'][i,good].max(),v['donor'][i].max())*1.13),xlabel='Body bearing (degrees)',ylabel='Structural distance (m)');ax.set_xticks([0,90,180,270,350]);ax.grid(alpha=.16)
  e=r['mae'];ax.set_title(f"{r['label']}\n{r['scene']} / query {r['query_id']} / seed 42",fontsize=11,pad=49)
  ax.text(0,1.04,f"MAE (m): original {e['normal']:.3f} | scalar {e['scalar']:.3f}\nrotated {e['rotated']:.3f} | donor {e['donor']:.3f} | angular r={r['correlation']:.3f}\nValid {r['valid']}/36; donor query {r['donor_query_id']}",transform=ax.transAxes,fontsize=10,linespacing=1.45)
 for j,name in enumerate(['queries_representative','queries_failure','queries_controls']):
  fig,axes=plt.subplots(1,2,figsize=(15,5.2));fig.subplots_adjust(top=.73,bottom=.21,wspace=.22)
  for ax,r in zip(axes,chosen[2*j:2*j+2]):panel(ax,r)
  h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.01));finish(fig,name)
 # Verify paper-style pooled-ray-per-scene contrast, not average query MAE.
 summaries={};official=read(a.docs/'official_contrasts.json')['contrasts'];summary_source=read(a.docs/'official_summary.json')
 for model in models:
  subset=[r for r in rows if r['model']==model];stats={}
  for k in ['scalar','rotated','donor']:
   means=[sum(r['delta'][k]*r['valid'] for r in subset if r['scene']==s)/sum(r['valid'] for r in subset if r['scene']==s) for s in scenes]
   value=float(np.mean(means));arch,head=model.split('_');contrast={'scalar':'scalarized_minus_normal','rotated':'rotated180_minus_normal','donor':'shuffle_minus_normal'}[k]
   expected=next(r for r in official if r['architecture']==arch and r['head']==head and r['input_condition']=='furnished' and r['contrast']==contrast)
   assert abs(value-expected['mean'])<2e-6,(model,k,value,expected['mean'])
   stats[k]=dict(scene_pooled_mean=value,original_ci95=expected['ci95'],scene_means=means,query_mean=float(np.mean([r['delta'][k] for r in subset])),query_median=float(np.median([r['delta'][k] for r in subset])),fraction_positive=float(np.mean([r['delta'][k]>0 for r in subset])))
  stats['angular_correlation_mean']=float(np.mean([r['correlation'] for r in subset if r['correlation'] is not None]));stats['prediction_angular_sd_median']=float(np.median([r['prediction_angular_sd'] for r in subset]));summaries[model]=stats
 fig,axs=plt.subplots(2,3,figsize=(15,9));fig.subplots_adjust(hspace=.48,wspace=.3,top=.88)
 values=[np.array([r['delta'][k] for r in primary]) for k in ['scalar','rotated','donor']];low=min(x.min() for x in values);high=max(x.max() for x in values);bins=np.linspace(low,high,70)
 for ax,k,x in zip(axs[0],['scalar','rotated','donor'],values):
  ax.hist(x,bins=bins,color=COL[k],alpha=.85);ax.axvline(0,color='#333',lw=1,ls='--');ax.set(xlim=(low,high),xlabel='Change in per-query MAE (m)',ylabel='Query × seed count',title={'scalar':'Scalarized − original','rotated':'Rotated 180° − original','donor':'Donor − original'}[k])
  ax.text(.03,.94,f"query mean {x.mean():+.4f} m\nmedian {np.median(x):+.4f} m\npositive: {np.mean(x>0):.1%}",transform=ax.transAxes,va='top',fontsize=9)
  if k!='donor':
   ins=inset_axes(ax,width='44%',height='40%',loc='upper right');ins.hist(x,bins=35,color=COL[k]);ins.axvline(0,color='#333',ls='--',lw=.7);ins.tick_params(labelsize=7);ins.set_title('Full local range (zoom)',fontsize=7)
 cc=np.array([r['correlation'] for r in primary if r['correlation'] is not None]);axs[1,0].hist(cc,bins=np.linspace(-1,1,31),color=COL['normal']);axs[1,0].axvline(0,color='#333',ls='--');axs[1,0].set(title='Within-query angular correlation',xlabel='Pearson r on valid directions',ylabel='Query × seed count');axs[1,0].text(.03,.94,f'mean {cc.mean():+.4f}; median {np.median(cc):+.4f}',transform=axs[1,0].transAxes,va='top',fontsize=9)
 sd=np.array([r['prediction_angular_sd'] for r in primary]);gs=np.array([r['gt_valid_angular_sd'] for r in primary]);b=np.geomspace(min(sd.min(),gs[gs>0].min())*.8,max(sd.max(),gs.max())*1.1,45)
 axs[1,1].hist(sd,bins=b,alpha=.8,color=COL['normal'],label='Prediction: all 36');axs[1,1].hist(gs,bins=b,histtype='step',lw=2,color=COL['gt'],label='GT: valid directions');axs[1,1].set(xscale='log',title='Angular standard deviation',xlabel='SD across directions (m; log axis)',ylabel='Query × seed count');axs[1,1].legend(fontsize=8)
 ax=axs[1,2]
 for j,k in enumerate(['scalar','rotated','donor']):
  v=summaries['spatial_regression'][k];ax.scatter(v['scene_means'],np.full(17,j),s=25,color=COL[k],alpha=.65);ax.plot(v['original_ci95'],[j+.18,j+.18],color=COL[k],lw=3);ax.plot(v['scene_pooled_mean'],j+.18,'D',color=COL[k],ms=5)
 ax.axvline(0,color='#333',ls='--');ax.set(yticks=[0,1,2],yticklabels=['Scalarized','Rotated','Donor'],xlabel='Scene-pooled MAE change (m)',title='17 scenes: dots; mean / 95% CI');ax.invert_yaxis()
 fig.suptitle('Spatial regression • furnished official Test • 17 scenes × 64 queries × 3 seeds\nQuery distributions are descriptive; uncertainty uses 17 source families, not 3,264 independent samples',fontsize=13);finish(fig,'aggregate_evidence')
 # Actual pair and within-scene scalar association, all from saved outputs.
 chosen_donor=chosen[4];scene=chosen_donor['scene'];i=chosen_donor['index'];v=cache[scene,42];j=int(v['donor_index'][i]);fig,axes=plt.subplots(1,3,figsize=(16,5));fig.subplots_adjust(wspace=.28,bottom=.22,top=.8)
 xx=[];yy=[]
 for scene0 in scenes:
  rr=[r for r in seed42 if r['scene']==scene0];x=np.array([r['gt_valid_mean'] for r in rr]);y=np.array([r['prediction_valid_mean'] for r in rr]);xx.extend(x-x.mean());yy.extend(y-y.mean())
 axes[0].scatter(xx,yy,s=10,alpha=.25,color=COL['normal']);axes[0].axhline(0,color='#aaa',lw=.7);axes[0].axvline(0,color='#aaa',lw=.7);axes[0].set(xlabel='GT mean − scene mean (m)',ylabel='Prediction mean − scene mean (m)',title=f'Position-dependent scale\nwithin-scene centred r={corr(np.array(xx),np.array(yy)):.3f}')
 for q,lab,col in [(i,'Query',COL['normal']),(j,'Donor position',COL['donor'])]:
  good=v['mask'][q];truth=np.where(good,v['gt'][q],np.nan);pred=np.where(good,v['normal'][q],np.nan)
  axes[1].plot(angles,truth,color=col,alpha=.5,lw=1,label=lab+' GT');axes[1].plot(angles,pred,color=col,lw=2.5,label=lab+' prediction')
  axes[2].plot(angles,truth-np.nanmean(truth),color=col,alpha=.5,lw=1);axes[2].plot(angles,pred-np.nanmean(pred),color=col,lw=2.5)
 for ax in axes[1:]:ax.set(xlabel='Body bearing (degrees)',xlim=(0,350));ax.set_xticks([0,90,180,270,350]);ax.grid(alpha=.15)
 axes[1].set(title='Same scene, two actual positions',ylabel='Distance (m)');axes[2].set(title='Remove each profile’s valid-direction mean',ylabel='Relative directional shape (m)');axes[2].axhline(0,color='#555',ls='--',lw=1)
 h,l=axes[1].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False)
 fig.suptitle(f'Range-scale evidence ≠ directional shape reconstruction\nExample: {scene}; queries {refs[scene][i]["query_id"]} and {refs[scene][j]["query_id"]}; seed 42',fontsize=13);finish(fig,'range_vs_shape')
 # Matched phase-zero trained counterpart; not an inference-only phase intervention.
 phase_stats=[];phase_cache={}
 for scene0 in scenes:
  for seed in [42,43,44]:
   z=load(a.study/'official_evaluation_v1/furnished/phase_zero_v1'/f'spatial_regression_{seed}'/f'{scene0}__normal.npz');v0=cache[scene0,seed];assert np.array_equal(z['mask'],v0['mask']) and np.array_equal(z['target'],v0['gt']);good=z['mask'];delta=float((np.abs(z['mean']-z['target'])[good].mean())-np.abs(v0['normal']-v0['gt'])[good].mean());phase_stats.append(delta)
   if seed==42:phase_cache[scene0]=z['mean']
 phase_mean=float(np.mean(phase_stats));expected=next(r for r in official if r['architecture']=='spatial' and r['head']=='regression' and r['input_condition']=='furnished' and r['contrast']=='phase_zero_minus_full');assert abs(phase_mean-expected['mean'])<2e-6
 fig,axes=plt.subplots(1,2,figsize=(14,4.8));fig.subplots_adjust(top=.72,bottom=.23,wspace=.23)
 for ax,r in zip(axes,chosen[:2]):
  v=cache[r['scene'],42];i=r['index'];good=v['mask'][i];pp=phase_cache[r['scene']][i];err=float(np.abs(pp-v['gt'][i])[good].mean())
  ax.plot(angles,np.where(good,v['gt'][i],np.nan),'.-',color=COL['gt'],label='GT');ax.plot(angles,v['normal'][i],lw=2,color=COL['normal'],label='Full-phase model');ax.plot(angles,pp,'--',lw=2,color=COL['phase'],label='Phase-zero trained model');ax.set(xlabel='Body bearing (degrees)',ylabel='Distance (m)',title=f"{r['scene']} / query {r['query_id']} / seed42\nMAE: full {r['mae']['normal']:.3f}; phase-zero {err:.3f} m",xlim=(0,350));ax.grid(alpha=.15)
 fig.suptitle(f'Matched phase control: phase-zero − full = {phase_mean:+.3f} m\n17-scene paired 95% CI [{expected["ci95"][0]:.3f}, {expected["ci95"][1]:.3f}]; identical selected queries',fontsize=13);h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False);finish(fig,'phase_control')
 result=dict(code_sha=a.code_sha,source_docs_commit='d52936cab06e96476ad4ead6ce79145c3b417e52',environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),scope='furnished official Test saved outputs; all4 primary models ×3seeds ×17scenes ×64queries',primary_query_seed_records=len(primary),raw_pairs_verified=verified,selection='seed42; >=18 valid directions; representative nearest40th/60th MAE percentile in distinct scenes; largest2 MAEs in distinct scenes; maximum mean absolute donor prediction change; minimum absolute scalar delta within0.3m of median MAE and GT SD>1m; unique queries',selected=chosen,summaries=summaries,phase_zero_minus_full=phase_mean,within_scene_centered_scalar_correlation_seed42=corr(np.array(xx),np.array(yy)),input_hashes=inputs,docs_hashes={n:sha(a.docs/n) for n in ['official_results.md','conclusion.md','results.md','representation_summary.json','official_summary.json','directional_diagnostic.json','official_contrasts.json','predefined.png','failure.png']})
 (a.out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','docs_hashes','selected']},indent=2))
if __name__=='__main__':main()
