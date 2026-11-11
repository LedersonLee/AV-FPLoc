"""Verify actual temperature curves against native field diagnostics and plot reliability."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from temperature_calibration import crossfit

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();base=read(a.baseline/'summary.json');cal=read(a.calibration/'summary.json');assert base['complete'] and cal['complete'] and base['queries']==cal['queries']==9965 and base['seed']==cal['seed'];temps=np.array(cal['temperatures']);t1=int(np.flatnonzero(temps==1)[0]);curves={};truth={};scenes=[];count=0;records=[]
 for rec in cal['records']:
  sid=rec['scene'];path=a.calibration/rec['file'];assert sha(path)==rec['sha256'];z=np.load(path);br=next(r for r in base['records'] if r['scene']==sid);bp=a.baseline/br['file'];assert sha(bp)==br['sha256'];rows=read(bp);np.testing.assert_array_equal(z['query_ids'],[r['query_id'] for r in rows]);scenes.extend([sid]*len(rows))
  for name in cal['methods']:
   values=z[name+'_curve'];expected=np.array([[r['methods'][name][k] for k in ['nearest_grid_nll','gt_region_nll','joint_confidence','entropy_normalized','gt_region_mass']] for r in rows]);np.testing.assert_allclose(values[:,t1],expected,atol=1e-9,rtol=1e-9);np.testing.assert_array_equal(z[name+'_top1'],[r['methods'][name]['top1_id'] for r in rows]);np.testing.assert_array_equal(z[name+'_joint'],[r['methods'][name]['joint'] for r in rows]);curves.setdefault(name,[]).append(values);truth.setdefault(name,[]).extend(z[name+'_joint'].tolist());count+=len(rows)
  records.append(dict(scene=sid,baseline_sha256=br['sha256'],curve_sha256=rec['sha256']))
 a.out.mkdir(parents=True,exist_ok=False);fig,axes=plt.subplots(len(curves),2,figsize=(11,4.3*len(curves)),squeeze=False);results={};scenes=np.array(scenes)
 for row,(name,parts) in enumerate(curves.items()):
  values=np.concatenate(parts);cf=crossfit(scenes,values,truth[name],temps);assert cf==cal['methods'][name];native=base['methods'][name];results[name]=dict(native_nearest_nll=native['means']['nearest_grid_nll'],crossfit_nearest_nll=cf['micro_nearest_grid_nll'],native_ece=native['joint_event_calibration']['ece'],crossfit_ece=cf['joint_calibration']['ece'],native_brier=native['joint_event_calibration']['brier'],crossfit_brier=cf['joint_calibration']['brier'],native_entropy=native['means']['entropy_normalized'],crossfit_entropy=cf['entropy_normalized'],joint_accuracy_unchanged=float(np.mean(truth[name])),fold_temperatures=[f['temperature'] for f in cf['folds']])
  ax=axes[row,0];ax.plot([0,1],[0,1],'--',color='gray',label='Ideal reliability')
  for label,detail,color in [('Native T=1',native['joint_event_calibration'],'#b44b40'),('Held-development temperature',cf['joint_calibration'],'#2472a4')]:
   bins=detail['bins'];ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',color=color,label=label)
  ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted joint-neighborhood mass',ylabel='Observed joint success',title=f'{name.upper()} seed{cal["seed"]}: reliability');ax.legend(fontsize=8);ax.grid(alpha=.2)
  ax=axes[row,1];loss=np.array([values[scenes==s,:,0].mean(0) for s in np.unique(scenes)]).mean(0);ax.plot(temps,loss,'o-',markersize=3,color='#2472a4');ax.axvline(1,color='#b44b40',linestyle='--',label='Native T=1')
  for i,t in enumerate(sorted(set(results[name]['fold_temperatures']))):ax.axvline(t,color='#548b54',alpha=.7,linestyle=':',label='Fit-fold selected T' if i==0 else None)
  ax.set(xscale='log',yscale='log',xlabel='Temperature',ylabel='Scene-mean nearest-grid NLL (log axis)',title='All-Val curve: descriptive, not selection');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle('Reused Val19 development | 19 scenes, 9,965 queries | Top1 poses unchanged',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.out/'reliability.png',dpi=160);plt.close(fig)
 (a.out/'verification.json').write_text(json.dumps(dict(status='pass',query_method_checks=count,scenes=19,results=results,sources=records,scope='Held-development temperature calibration; prior training/checkpoint/teacher exposure persists. ECE residual is not zero. No independent calibration claim.',source_sha256=sha(__file__),baseline_summary_sha256=sha(a.baseline/'summary.json'),calibration_summary_sha256=sha(a.calibration/'summary.json')),indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
