from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'figures';D.mkdir(exist_ok=True)
a=json.loads((P/'analysis.json').read_text());e=json.loads((P/'extension_analysis.json').read_text())
plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig,ax=plt.subplots(figsize=(7,4));k=[4,8,12,36];ax.plot(k,[48.893,92.122,93.164,96.289],'o-',label='Full bank');ax.plot(k,[69.206,79.102,80.143,83.919],'s-',label='Fixed visual Top20');ax.set(xlabel='Oracle ray directions',ylabel='Joint recall (%)',ylim=(40,100),title='Geometry capacity is not acoustic predictability');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(D/'oracle_capacity.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for kind in ['ecn','embedding','raw','chirp']:
 axes[0].plot(k,[a['ray_summary'][f'acoustic/{kind}/K{x}/subset{x}']['mae']['mean'] for x in k],'o-',label=kind)
 axes[1].plot(k,[100*a['summary'][f'acoustic/{kind}/K{x}']['joint']['mean'] for x in k],'o-',label=kind)
axes[0].set(ylabel='Ray MAE (m)',title='Different K targets; do not infer precision alone');axes[1].set(ylabel='Joint recall (%)',title='Same fixed visual candidate bank')
for ax in axes:ax.set_xlabel('Trained ray directions');ax.grid(alpha=.2)
axes[0].legend();fig.tight_layout();fig.savefig(D/'acoustic_representation.png');plt.close(fig)
names=list(a['contrasts']);means=[100*a['contrasts'][n]['joint']['mean'] for n in names];bounds=np.array([a['contrasts'][n]['joint']['ci95'] for n in names])*100
fig,ax=plt.subplots(figsize=(8,4));ax.errorbar(means,np.arange(len(names)),xerr=np.stack([np.array(means)-bounds[:,0],bounds[:,1]-means]),fmt='o',capsize=3);ax.set_yticks(np.arange(len(names)),names);ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Joint recall difference (percentage points)',title='New12 paired scene bootstrap; exploratory 95% CIs');fig.tight_layout();fig.savefig(D/'joint_contrasts.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));arms=['acoustic/raw','acoustic/chirp','acoustic/embedding','joint/attention','joint/concat','joint/disabled','joint/late'];x=np.arange(len(arms));new=[a['summary'][n+'/K36']['joint']['mean']*100 for n in arms];tra=[e['transfer'][n+'/matched/uniform/visual_geometry']['joint']['mean']*100 for n in arms];ax.bar(x-.18,new,.36,label='New12');ax.bar(x+.18,tra,.36,label='Transfer11');ax.axhline(100*e['transfer_baselines']['GEOQ']['joint']['mean'],color='black',ls='--',label='Transfer B2+GEOQ (its own candidates)');ax.set_xticks(x,[n.split('/')[-1] for n in arms],rotation=25);ax.set(ylabel='Joint recall (%)',ylim=(40,80),title='Ray methods remain below the strong framework baseline');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(D/'transfer.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.6));cal=json.loads((P/'calibration_results.json').read_text())['records'];names=sorted(set(r['group']+'/'+r['kind'] for r in cal))
for i,n in enumerate(names):
 rr=[r for r in cal if r['group']+'/'+r['kind']==n];axes[0].bar(i,100*np.mean([r['interval90_coverage'] for r in rr]));axes[1].scatter(np.mean([r['predicted_mean_absolute_log_error'] for r in rr]),np.mean([r['mean_absolute_log_error'] for r in rr]),label=n)
axes[0].axhline(90,color='black',ls='--');axes[0].set_xticks(range(len(names)),[n.split('/')[-1] for n in names],rotation=20);axes[0].set(ylabel='Held residual interval coverage (%)',ylim=(80,100));axes[1].plot([.25,.4],[.25,.4],ls='--',c='gray');axes[1].set(xlabel='Inner-OOF mean absolute log residual',ylabel='Outer-held actual residual');axes[1].legend(fontsize=7);fig.suptitle('Global directional calibration; not query-specific confidence');fig.tight_layout();fig.savefig(D/'calibration.png');plt.close(fig)
