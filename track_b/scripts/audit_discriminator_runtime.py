"""Head-only runtime replay: numeric features in, candidate choice out."""
import argparse,builtins,hashlib,inspect,json
from pathlib import Path
import torch
from track_b.models.discrimination import GeometryDiscriminator

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def infer(model,features,mean,std,arm):
 assert features.ndim==3 and features.shape[-1]==133 and torch.isfinite(features).all()
 assert mean.shape==std.shape==(133,) and (std>0).all()
 assert arm in ['geometry','score_statistics_only']
 x=(features-mean)/std
 if arm=='score_statistics_only':x=x.clone();x[...,:128]=0
 return model(x).argmax(-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
 run=a.root/'discriminator_v1';candidate_dir=a.root/'development_candidates_v1'
 records=[];candidate_hashes={}
 for path in sorted(candidate_dir.glob('*.pt')):records.extend(torch.load(path,weights_only=True,map_location='cpu'));candidate_hashes[path.name]=sha(path)
 historical=read(run/'predictions.json');fits=read(run/'fits.json');audits=[]
 for fit in fits:
  path=run/f"{fit['run_id']}.pt";assert sha(path)==fit['checkpoint_sha256'];saved=torch.load(path,weights_only=True,map_location='cpu');model=GeometryDiscriminator().eval();model.load_state_dict(saved['model'])
  selected=[r for r in records if r['seed']==fit['seed'] and r['policy']==fit['policy']];features=torch.stack([r['features'] for r in selected]);assert len(selected)==384
  # Warm up before monitoring; no identifiers, labels or poses enter infer.
  infer(model,features[:1],saved['mean'],saved['std'],saved['arm'])
  calls=[];original=builtins.open
  def observed_open(*args,**kwargs):calls.append(str(args[0]));return original(*args,**kwargs)
  builtins.open=observed_open
  try:indices=infer(model,features,saved['mean'],saved['std'],saved['arm'])
  finally:builtins.open=original
  assert not calls,'Python-level file access during head inference'
  expected={(r['scene'],r['query']):r for r in historical if r['seed']==fit['seed'] and r['policy']==fit['policy'] and r['arm']==fit['arm'] and r['method']==fit['arm']}
  assert len(expected)==len(selected)
  for r,k in zip(selected,indices.tolist()):
   # Labels are consulted only after inference, for historical metric replay.
   old=expected[r['scene'],r['query']]
   actual=dict(joint=float(r['labels'][k]),r1=float(r['distance'][k]<=1+1e-6),fine=float(r['distance'][k]<=.1+1e-6),translation_error_m=float(r['distance'][k]),yaw_error_deg=float(r['yaw'][k]*180/torch.pi))
   assert all(abs(old[key]-value)<1e-5 for key,value in actual.items()),(fit['run_id'],r['scene'],r['query'])
  audits.append(dict(run_id=fit['run_id'],checkpoint_sha256=sha(path),queries=len(selected),python_open_calls=calls,metric_mismatches=0))
 a.out.mkdir(parents=True,exist_ok=False)
 (a.out/'validation.json').write_text(json.dumps(dict(code_sha=a.code_sha,fits=len(audits),query_decisions=sum(x['queries'] for x in audits),input_signature=str(inspect.signature(infer)),candidate_hashes=candidate_hashes,predictions_sha256=sha(run/'predictions.json'),audits=audits,scope='Cached-feature discriminator head only; no raw observation or candidate builder replay; Python open hook does not trace native I/O',test_accessed=False),indent=2));print('COMPLETE',len(audits),flush=True)
if __name__=='__main__':main()
