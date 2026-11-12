"""Measured completed-run costs, explicitly not isolated hardware benchmarks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from model import AVLoc


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();records=[]
 for arm in ['A0','A1','A2']:
  run=a.study/f'runs/{arm}_seed42';complete=json.loads((run/'complete.json').read_text());assert complete['status']=='training_complete'
  best=json.loads((run/'best.json').read_text());protocol=json.loads((run/'protocol.json').read_text());files=sorted(run.glob('epoch_*.json'));epochs=[json.loads(p.read_text()) for p in files]
  assert [r['epoch'] for r in epochs]==list(range(1,len(epochs)+1))
  assert all(r['train_queries']==58894 for r in epochs)
  ck=torch.load(run/best['path'],map_location='cpu',weights_only=False);assert ck['epoch']==best['epoch']
  model=AVLoc(arm);model.load_state_dict(ck['model'],strict=True);parameter_count=sum(p.numel() for p in model.parameters());del model
  state=ck['model'];seconds=np.array([r['seconds'] for r in epochs]);params={k:v for k,v in state.items() if not any(k.endswith(x) for x in ['running_mean','running_var','num_batches_tracked'])}
  # State tensors exclude nonpersistent buffers; parameter proxy is checked against model in later runtime profiling.
  records.append({'arm':arm,'seed':42,'executed_epochs':len(epochs),'receipt_last_epoch':complete['last_epoch'],'best_epoch':best['epoch'],'reason':complete['reason'],
   'epoch_seconds_sum':float(seconds.sum()),'epoch_seconds_median':float(np.median(seconds)),'epoch_seconds_min':float(seconds.min()),'epoch_seconds_max':float(seconds.max()),
   'training_sample_visits':sum(r['train_queries'] for r in epochs),'optimizer_updates':sum(r['optimizer_updates'] for r in epochs),
   'best_checkpoint_file_bytes':(run/best['path']).stat().st_size,'state_tensor_elements':sum(v.numel() for v in state.values()),'parameter_like_elements_excluding_bn_buffers':sum(v.numel() for v in params.values()),'verified_model_parameters':parameter_count,
   'state_tensor_payload_bytes':sum(v.numel()*v.element_size() for v in state.values()),'checkpoint_sha256':sha(run/best['path']),
   'training_code_commit':protocol['code_commit'],'protocol_sha256':sha(run/'protocol.json'),'epoch_record_sha256':{f.name:sha(f) for f in files}})
  del ck,state,params
 result={'complete':True,'records':records,'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['audit_completed_training_cost.py','model.py']},'test_accessed':False,
 'scope':'Recorded epoch wall time includes training and validation, excludes initial validation/startup, and overlaps other jobs. Not isolated GPU compute or end-to-end latency. Checkpoint file includes metadata; tensor payload separate. Parameter-like count is a state-dict accounting proxy, not FLOPs.',
 'limitations':['No historical FP-RIR rendering wall time recovered.','No new training or checkpoint modification.','Single seed; data loading/caching/concurrent jobs affect elapsed time.','Pending A3 and interaction-disabled training excluded.','Recorded complete.last_epoch can exceed actual executed epochs on patience break; executed epoch files are authoritative.']}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps([{k:r[k] for k in ['arm','executed_epochs','epoch_seconds_sum','epoch_seconds_median','parameter_like_elements_excluding_bn_buffers','state_tensor_payload_bytes','best_checkpoint_file_bytes','verified_model_parameters']} for r in records],indent=2))
if __name__=='__main__':main()
