"""Deterministically display full fields for positive and negative development cases."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from track_b.models.interaction import InteractionMatcher
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.scripts.fusion_photometric import descriptors
from track_b.eval.metrics import pose_statistics
from track_b.scripts.train_structure import diverse_cpu

def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
 att={(r['scene'],r['query']):r for r in read(a.run/'attention_42.json')['val']['rows'] if r['arm']=='normal'};fixed={(r['scene'],r['query']):r for r in read(a.prior/'independent_fusion_v1/predictions.json') if r['method']=='fixed' and r['seed']==42};groups={name:[] for name in ('fixed_correct_attention_wrong','attention_correct_fixed_wrong','both_wrong_attention_raw20_hit','both_wrong_attention_raw20_miss')}
 for key in sorted(att):
  x=att[key];y=fixed[key]
  if y['joint'] and not x['joint']:groups['fixed_correct_attention_wrong'].append(key)
  elif x['joint'] and not y['joint']:groups['attention_correct_fixed_wrong'].append(key)
  elif not x['joint'] and not y['joint']:groups['both_wrong_attention_raw20_hit' if x['raw20'] else 'both_wrong_attention_raw20_miss'].append(key)
 models={}
 for mode in ('attention','concat','separable'):
  m=InteractionMatcher(mode).eval();m.load_state_dict(torch.load(a.run/f'{mode}_42.pt',weights_only=True,map_location='cpu'));models[mode]=m
 independent={}
 for mode in ('visual','acoustic'):
  m=StructureMatcher().eval();m.load_state_dict(torch.load(a.prior/f'supervised_v1/{mode}_42.pt',weights_only=True,map_location='cpu'));independent[mode]=m
 records=[]
 with torch.no_grad():
  for category,keys in groups.items():
   if not keys:continue
   sid,q=keys[0];s=torch.load(a.cache/f'{sid}.pt',weights_only=True,map_location='cpu');idx=s['query_ids'].index(q);rgb=s['rgb'][idx:idx+1].float()/255;spec=s['spectrum'][idx:idx+1].float();cal={'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']};poses=s['candidates'][:,:3];gt=s['gt'][idx];fields={mode:m(rgb,spec,s['floorplan'],cal,s['candidates'])[0] for mode,m in models.items()};w=.25+.75*s['candidates'][:,19:35];parts={}
   for mode,m in independent.items():
    fp=descriptors(m,s);obs=m.visual(rgb) if mode=='visual' else m.acoustic(spec);parts[mode]=m.match(obs,fp,w*visible_rays(s['hfov_deg']) if mode=='visual' else w)[0]
   fields['independent_fixed']=.85*parts['visual']+.15*parts['acoustic'];fig,axes=plt.subplots(2,4,figsize=(18,7),constrained_layout=True);metrics={};xy=poses[::36,:2];near=((xy-gt[:2]).square().sum(-1)).argmin();origin=s['origin'].tolist();free=s['floorplan'][0,0].numpy();extent=[origin[0],origin[0]+free.shape[1]*s['spacing'],origin[1],origin[1]+free.shape[0]*s['spacing']]
   for col,(mode,field) in enumerate(fields.items()):
    metric=pose_statistics(field,poses,gt,diverse=True);metrics[mode]=metric;raw=field.topk(20).indices;div=diverse_cpu(field.numpy(),poses.numpy());best=field.argmax();mass=field.reshape(-1,36).logsumexp(-1);mass=mass-mass.max();ax=axes[0,col];ax.imshow(free,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1,alpha=.3);im=ax.scatter(xy[:,0],xy[:,1],c=mass.numpy(),s=10,marker='s',cmap='viridis',vmin=-8,vmax=0);ax.scatter(poses[raw,0],poses[raw,1],s=22,facecolors='none',edgecolors='cyan',label='raw20');ax.scatter(poses[div,0],poses[div,1],marker='+',s=30,c='white',label='diverse20');ax.scatter(gt[0],gt[1],marker='*',s=130,c='red',label='GT');ax.scatter(poses[best,0],poses[best,1],marker='x',s=80,c='orange',label='MAP');ax.set_aspect('equal');ax.set_title(f"{mode}\nJoint={int(metric['joint'])}, Raw20={int(metric['raw20'])}, Diverse20={int(metric['diverse20'])}");ax.set_xlabel('native world X (m)');ax.set_ylabel('native world Z (m)')
    if col==0:ax.legend(fontsize=7,loc='best')
    curve=field.reshape(-1,36)[near].softmax(-1);axes[1,col].plot(np.arange(36)*10,curve.numpy());axes[1,col].axvline(float(gt[2]*180/math.pi)%360,color='red',linestyle='--');axes[1,col].set(xlabel='native yaw (degrees)',ylabel='conditional yaw probability',title='Yaw at nearest XY to GT (diagnostic only)',xlim=(0,360))
   fig.colorbar(im,ax=list(axes[0]),label='XY log-mass minus maximum (clipped at -8)');fig.suptitle(f'{category} | seed42 | {sid} / query{q}\nFirst lexicographic case under original GPU metrics; panels replay on CPU',fontsize=12);path=a.out/f'{category}.png';fig.savefig(path,dpi=135);plt.close(fig);records.append(dict(category=category,scene=sid,query=q,selection_gpu_attention=att[sid,q],selection_gpu_fixed=fixed[sid,q],cpu_metrics=metrics,image=path.name));print(category,flush=True)
 result=dict(selection='first scene/query in lexicographic order within each declared category, seed42 only; no image-quality selection',category_counts={k:len(v) for k,v in groups.items()},cases=records,not_available={'GT_absent_bank':'0 under audited original bank','physical_furniture_shift':'no paired condition in selected observations','false_physical_echo':'not physically identified'},limits='Exploratory illustration, not performance evidence; CPU replay may differ from original GPU near ties; yaw plot uses GT solely to select diagnostic spatial slice.')
 (a.out/'selection.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
