"""Replay every new checkpoint against its saved held/OOF predictions."""
from pathlib import Path
import json,hashlib,time
import numpy as np,torch
from torch import nn
from train_acoustic import RayModel
from joint_models import JointRay
P=Path('/output'); OLD=Path('/workspace/experiments/directional_tribranch_20261109')
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.2)
 pr=json.loads((P/'PROTOCOL.json').read_text()); allowed=set(pr['training_scenes']+pr['transfer_scenes']);data={s:dict(np.load(P/'audio'/s/'features.npz')) for s in allowed};obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in allowed};rows=[]
 cps=sorted(p for root in ['acoustic','joint','joint_followup','calibration','transfer_models'] for p in (P/root).rglob('*.pt'))
 for cp in cps:
  ck=torch.load(cp,map_location='cpu',weights_only=False);kind=ck['kind'];group=ck.get('group','joint' if cp.parts[-3] in ['joint','joint_followup'] else 'acoustic');cal=cp.parts[-3]=='calibration';transfer=cp.parts[-3]=='transfer_models'
  joint=group=='joint';m=JointRay('concat' if kind=='concat_wide' else kind) if joint else RayModel(kind,ck.get('k',36))
  if kind=='concat_wide':m.mixer=nn.Sequential(nn.Flatten(),nn.Linear(56*32,64),nn.GELU(),nn.Linear(64,32))
  m.load_state_dict(ck['state']);m.cuda().eval();held=ck.get('validation_scenes',ck.get('held_scenes',pr['transfer_scenes'] if transfer else []));fit=ck['fit_scenes'];assert set(held).isdisjoint(fit);assert set(fit+held)<=allowed
  mu=ck.get('mu',ck.get('mean',ck.get('am')));std=ck.get('std',ck.get('ast'));vm=ck.get('vm');vs=ck.get('vs',ck.get('vst'));maximum=0.;arrays=0
  def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
  with torch.inference_mode():
   for sid in held:
    saved=np.load(cp.parent/(sid+('_oof' if cal else '')+'.npz'));assert np.array_equal(saved['query_ids'],obs[sid]['query_ids']);key='raw' if joint else kind;x=tensor(np.clip((data[sid][key]-mu)/std,-10,10));v=tensor(np.clip((obs[sid]['visual']-vm)/vs,-10,10)) if joint else None
    outputs={'prediction':m(x,v).cpu().numpy() if joint else m(x).cpu().numpy()}
    if joint and 'donor' in saved:outputs['donor']=m(torch.roll(x,17,0),v).cpu().numpy()
    if not joint:
     for control in ['permuted','isotropic','sorted']:
      if control in saved:outputs[control]=m(tensor(np.clip((data[sid][control]-mu)/std,-10,10))).cpu().numpy()
    for key,pred in outputs.items():
     error=float(np.max(abs(pred-saved[key])));maximum=max(maximum,error);assert np.allclose(pred,saved[key],rtol=1e-5,atol=1e-6),(str(cp),sid,key,error);arrays+=1
  digest=hashlib.sha256(cp.read_bytes()).hexdigest()
  if not cal:assert digest==json.loads((cp.parent/'receipt.json').read_text())['checkpoint_sha256']
  else:
   inner=int(cp.stem[-1]);assert digest==json.loads((cp.parent/'complete.json').read_text())['inner_fits'][inner]['sha256']
  rows.append(dict(checkpoint=str(cp.relative_to(P)),sha256=digest,held_scenes=held,arrays=arrays,max_absolute_error=maximum));del m
  if len(rows)%12==0:print('REPLAY',len(rows),'/',len(cps),flush=True)
 assert len(rows)==228,len(rows)
 (P/'checkpoint_verification.json').write_text(json.dumps(dict(checkpoints=len(rows),arrays=sum(r['arrays'] for r in rows),max_absolute_error=max(r['max_absolute_error'] for r in rows),split_disjoint=True,query_ids_equal=True,receipt_hashes_equal=True,records=rows),indent=2));print('ALL REPLAY PASSED',flush=True)
if __name__=='__main__':main()
