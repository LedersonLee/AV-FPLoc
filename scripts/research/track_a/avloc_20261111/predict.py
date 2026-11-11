"""Extract every clean Val19 ray prediction from a completed run's selected checkpoint."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import AVLoc
from dataset import Observations
from augment import prepare

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 assert (a.run/'complete.json').is_file(),'Only completed training runs; no intermediate checkpoint selection'
 best=json.loads((a.run/'best.json').read_text());checkpoint=a.run/best['path'];ck=torch.load(checkpoint,map_location='cpu',weights_only=False);protocol=ck['protocol']
 assert sha(a.data/'manifest.json')==protocol['input_manifest_sha256']
 # Do not silently evaluate a checkpoint with a modified architecture implementation.
 assert sha(Path(__file__).with_name('model.py'))==protocol['source_sha256']['model.py']
 ds=Observations(a.data,'val');assert len(ds.records)==19 and len(ds)==9965
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=AVLoc(protocol['arm']).cuda().eval();model.load_state_dict(ck['model']);del ck
 a.out.mkdir(parents=True,exist_ok=False);records=[];storage={};started=time.time()
 with torch.inference_mode():
  for batch in DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True):
   rgb,spec,_=prepare(batch,'cuda',False);out=model(rgb,spec);arrays={}
   for key,value in out.items():
    if key=='weights':arrays['weights']=value.cpu().numpy()
    else:arrays[key+'_mu']=value[0].cpu().numpy();arrays[key+'_scale']=value[1].cpu().numpy()
   for i,sid in enumerate(batch['scene']):
    dest=storage.setdefault(sid,{'query_ids':[]});dest['query_ids'].append(int(batch['query_id'][i]))
    for key,value in arrays.items():dest.setdefault(key,[]).append(value[i])
 for r in ds.records:
  sid=r['scene'];arrays={k:np.asarray(v) for k,v in storage[sid].items()};assert len(arrays['query_ids'])==r['queries']
  labels=np.load(a.data/sid/'labels.npz');assert np.array_equal(arrays['query_ids'],labels['query_ids'])
  assert all(np.isfinite(v).all() for v in arrays.values());path=a.out/f'{sid}.npz';np.savez_compressed(path,**arrays);records.append(dict(scene=sid,queries=r['queries'],file=path.name,sha256=sha(path)))
 receipt=dict(complete=True,split='val',scenes=19,queries=len(ds),records=records,arm=protocol['arm'],seed=protocol['seed'],epoch=best['epoch'],checkpoint_sha256=sha(checkpoint),training_code_commit=protocol['code_commit'],extractor_sha256=sha(__file__),input_manifest_sha256=sha(a.data/'manifest.json'),precision='float32',selection='training best clean Val19 ray MAE; reused development',test_accessed=False,seconds=time.time()-started)
 (a.out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PREDICTIONS COMPLETE',len(ds),flush=True)
if __name__=='__main__':main()
