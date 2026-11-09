"""Compare identical short fits differing only in CPU/GPU scene residency."""
import argparse,json,hashlib
from pathlib import Path
import torch

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 dirs={k:a.root/f'residency_{k}_smoke_v1' for k in ['gpu','cpu']}
 protocols={k:read(d/'protocol.json') for k,d in dirs.items()}
 for k in dirs:assert protocols[k].pop('scene_residency')==k
 assert protocols['gpu']==protocols['cpu']
 states={k:torch.load(d/'visual_42.pt',weights_only=True,map_location='cpu') for k,d in dirs.items()}
 equal=all(torch.equal(states['gpu'][k],states['cpu'][k]) for k in states['gpu'])
 delta=max(float((states['gpu'][k]-states['cpu'][k]).abs().max()) for k in states['gpu'])
 records={k:read(d/'visual_42.json') for k,d in dirs.items()};predictions_equal=all(records['gpu'][part]['rows']==records['cpu'][part]['rows'] for part in ['train','val'])
 losses_equal=all({k:v for k,v in x.items() if k!='elapsed_s'}=={k:v for k,v in y.items() if k!='elapsed_s'} for x,y in zip(records['gpu']['history'],records['cpu']['history']))
 result=dict(parameter_bitwise_equal=equal,max_parameter_difference=delta,all_prediction_rows_equal=predictions_equal,losses_and_gradients_equal=losses_equal,compared_train_queries=1024,compared_val_queries=384,scope='same foundation visual seed42, one aux and one dense update, full16/6 cached cohort; no full-convergence equivalence claim',checkpoint_hashes={k:sha(d/'visual_42.pt') for k,d in dirs.items()})
 a.out.write_text(json.dumps(result,indent=2));assert equal and predictions_equal and losses_equal,result;print(json.dumps(result));print('COMPLETE')
if __name__=='__main__':main()
