"""Publication-style local figures from saved results, no invented examples."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from scipy.io import wavfile
from model import softmax

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);o=ap.parse_args().out;c=json.loads((o/'config.json').read_text());r=Path(c['root']);sid=c['scene'];figdir=o/'figures';figdir.mkdir(exist_ok=True);regs=json.loads((r/'source/hssd-hab/semantics/scenes'/(sid+'.semantic_config.json')).read_text())['region_annotations'];meta=json.loads((r/'2DFP'/sid/'floorplan_metadata.json').read_text());mp=np.array(Image.open(r/'2DFP'/sid/'map.png'));b=meta['world_bounds_m'];s=meta['map_spacing_m'];extent=[b['x_min']-s/2,b['x_min']+(mp.shape[1]-.5)*s,b['z_min']+(mp.shape[0]-.5)*s,b['z_min']-s/2];d=np.load(o/'inference_candidates.npz');gt=np.load(o/'evaluation_truth.npz')['truth'];pred=np.load(o/'oracle_predictions.npz');records=json.loads((o/'oracle_observations.json').read_text());images=json.loads((o/'images.json').read_text());metrics=json.loads((o/'metrics.json').read_text());pairs=json.loads((o/'acoustic_pairs.json').read_text());palette=plt.cm.tab10(np.arange(9))
 def save(fig,name):fig.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
 def floor(ax):
  ax.imshow(mp,cmap='gray',extent=extent,vmin=0,vmax=255)
  for i,reg in enumerate(regs):
   p=np.array(reg['poly_loop'])[:,[0,2]];ax.add_patch(Polygon(p,facecolor=palette[i],edgecolor=palette[i],alpha=.22));center=p.mean(0);ax.text(*center,f'{i}: {reg["label"]}',ha='center',fontsize=6,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
  ax.set_aspect('equal');ax.set(xlabel='World X (m)',ylabel='World Z (m), +Z down')
 def pose(ax,p,color,label):
  ax.scatter(*p[:2],c=color,s=35,label=label);v=.7*np.array([np.cos(np.deg2rad(p[2])),np.sin(np.deg2rad(p[2]))]);ax.arrow(*p[:2],*v,color=color,head_width=.15,length_includes_head=True)
 fig,ax=plt.subplots(figsize=(9,8));floor(ax)
 for i,p in enumerate(pairs):
  ax.scatter(p['source'][0],p['source'][2],marker='*',s=120,label=f'RIR source {i}');pose(ax,np.r_[np.array(p['center'])[[0,2]],0],'black',f'RIR receiver {i}')
 ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));ax.set_title(f'{sid} | native HSSD annotations + canonical structural FP\nThree acoustic pairs, geometry seed {c["geometry_seed"]}; source GT shown for generation only');save(fig,'semantic_overlay')
 cases=[]
 correct=pred['correct'];pout=pred['posterior'];n=len(gt)
 for typ,mask in [('success',correct[:,3]&~correct[:,0]),('failure',~correct[:,3]&correct[:,0])]:
  si,q=np.argwhere(mask)[0];rec=records[int(si)*n+int(q)];p=pout[si,3,q];v=pout[si,0,q];base=int(v.argmax());chosen=int(p.argmax());valid=np.flatnonzero(pred['valid'][q]);truthcandidate=int(valid[0]) if len(valid) else int(np.argmin(np.linalg.norm(d['poses'][q,:,:2]-gt[q,:2],axis=1)));show=list(dict.fromkeys([base,chosen,truthcandidate]));fig,axs=plt.subplots(1,3,figsize=(17,5),gridspec_kw={'width_ratios':[1,1.1,1.2]});axs[0].imshow(Image.open(images[str(rec['query_id'])]));axs[0].axis('off');axs[0].set_title('Actual cached RGB query');floor(axs[1]);pose(axs[1],gt[q],'green','GT robot')
  for k,color in zip(show,['red','blue','purple']):pose(axs[1],d['poses'][q,k],color,f'candidate {k}')
  src=rec['source_xyz'];axs[1].scatter(src[0],src[2],marker='*',s=120,c='orange',label='GT source (display only)');axs[1].legend(fontsize=7);x=np.arange(len(show));axs[2].bar(x-.16,v[show],width=.32,label='Visual mass');axs[2].bar(x+.16,p[show],width=.32,label='Fused mass');axs[2].set(xticks=x,xticklabels=[f'k{k}\nscore {d["scores"][q,k]:.2f}\nLdir relative {(p[k]/v[k])/(p/v).max():.3f}' for k in show],ylabel='Conditional Top-20 mass',ylim=(0,max(max(p[show]),max(v[show]))*1.25));axs[2].legend(fontsize=8);fig.suptitle(f'{typ.upper()} | {sid}, seed {rec["seed"]}, query {rec["query_id"]}\n{rec["event"]}, source room {rec["source_room"]}; joint<=1m &30deg; source never enters inference',fontsize=11);fig.tight_layout();save(fig,'case_'+typ);cases.append(dict(type=typ,seed=rec['seed'],query=rec['query_id'],q=int(q),selection='first qualifying seed/query, descriptive not representative frequency',image=images[str(rec['query_id'])],visual_top=base,fused_top=chosen,valid_candidates=valid.tolist(),fp_line_clear=rec['fp_line_clear']))
 fig,ax=plt.subplots(figsize=(8,4));noise=[0,15,30,60];names=['B3','B3_noise15','B3_noise30','B3_noise60'];m={x['method']:x for x in metrics['metrics']};ax.plot(noise,[m[x]['joint_top1_percent'] for x in names],'o-',label='Class + noisy oracle bearing');ax.axhline(m['B0']['joint_top1_percent'],color='gray',ls='--',label='Visual only');ax.set(xlabel='Injected Gaussian angular noise SD (degrees)',ylabel='Joint Top-1 (%)',xticks=noise,title=f'{sid} | 124 queries x seeds 42/43/44\nFixed 15-degree likelihood kernel, no tuning');ax.legend();save(fig,'noise_sensitivity')
 audio=json.loads((o/'audio_validation.json').read_text());fig,axs=plt.subplots(2,2,figsize=(12,9));a=np.deg2rad(c['mic_clockwise_angles_deg']);xy=.05*np.stack([np.cos(a),np.sin(a)],1);axs[0,0].scatter(*xy.T)
 for i,p in enumerate(xy):axs[0,0].text(*p,f'raw {i}',fontsize=9)
 axs[0,0].arrow(0,0,.065,0,head_width=.003,color='green');axs[0,0].set(xlabel='Body forward (m)',ylabel='Body right (m)',title='Six selected UMA raw channels; radius provisional');axs[0,0].set_aspect('equal')
 for pair in [0,1]:
  for heading,marker in [(0,'o'),(90,'s')]:
   row=next(x for x in audio['rows'] if x['furnished'] and x['seed']==42 and x['pair']==pair and x['heading']==heading);t=np.array(row['onset_samples']);e=np.array(row['expected_samples']);axs[0,1].plot(e-e[0],t-t[0],marker,label=f'pair{pair} h{heading}')
 axs[0,1].plot([-15,15],[-15,15],'k--');axs[0,1].set(xlabel='Geometric TDOA vs ch0 (samples)',ylabel='Observed onset difference (samples)',title='LOS conditions: max residual 0.73 sample');axs[0,1].legend(fontsize=8)
 arr=np.load(o/'rirs/furnished_p0_h0_s42.npy')
 for i,ch in enumerate(arr):axs[1,0].plot(np.arange(130,220)/48,ch[130:220]+i*.35,label=f'ch{i}')
 axs[1,0].set(xlabel='Time (ms)',ylabel='RIR (vertical display offsets)',title='Bathroom p0/h0/seed42: actual mono RIRs')
 for role in ['furnished','stage']:
  arr=np.load(o/'direct_probe'/f'{role}_p2_h0.npy');axs[1,1].plot(np.arange(arr.shape[1])/48,arr[0],label=role)
 axs[1,1].set(xlim=(9,15),xlabel='Time (ms)',ylabel='Direct-only RIR ch0',title='Living-room p2: furniture removes direct path');axs[1,1].legend();fig.suptitle(f'{sid} | physical receiver-coordinate audit; not a point-mic certification',fontsize=12);fig.tight_layout();save(fig,'microphone_timing')
 sr,wav=wavfile.read(o/'broadband_received_6ch.wav');fig,axs=plt.subplots(3,2,figsize=(11,8))
 for k,ax in enumerate(axs.ravel()):ax.specgram(wav[:,k],Fs=sr,NFFT=256,noverlap=192);ax.set(ylim=(0,12000),title=f'Raw channel {k}',xlabel='Time (s)',ylabel='Hz')
 fig.suptitle(f'{sid} p0/h0/seed42 | synthetic broadband convolved with measured simulator RIR\nNot a household sound recording');fig.tight_layout();save(fig,'broadband_spectrogram');(o/'figure_cases.json').write_text(json.dumps(cases,indent=2))
if __name__=='__main__':main()
