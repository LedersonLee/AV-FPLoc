"""Verify complete factorial inference controls and pair each planned contrast."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['run','training','reference','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();setup=read(a.run/'protocol.json');training=read(a.training/'protocol.json');ref=read(a.reference/'protocol.json')
 assert not setup['smoke'] and setup['source_protocol_sha256']==sha(a.training/'protocol.json')
 assert training['encoder_regime']=='frozen' and setup['reference_protocol_sha256']==sha(a.reference/'protocol.json')
 assert training['pretrained_sha256']==ref['checkpoint_hashes'] and training['data']['scenes']['val']==ref['scene_ids']
 assert read(a.run/'normal_replay_mismatches.json')==read(a.run/'initial_replay_mismatches.json')==[]
 for mode in training['modes']:
  for seed in training['seeds']:
   for kind in ['initial','final']:assert setup['checkpoint_hashes'][f'{mode}_{seed}_{kind}']==sha(a.training/(f'{mode}_{seed}'+('_initial' if kind=='initial' else '')+'.pt'))
 rows=read(a.run/'predictions.json');key=lambda r:tuple(r[k] for k in ['mode','seed','scene','query','condition'])
 assert len(rows)==len({key(r) for r in rows})==13824
 expected={(r['seed'],r['scene'],r['query']) for r in read(a.reference/'predictions.json') if r['method']=='fixed'};assert len(expected)==1152
 for mode in training['modes']:
  for condition in setup['conditions']:
   assert {(r['seed'],r['scene'],r['query']) for r in rows if r['mode']==mode and r['condition']==condition}==expected
 contrasts=[('initial_scales','normal','scale_restoration_with_interaction'),('both','zero_interaction','scale_restoration_without_interaction'),('zero_interaction','normal','remove_interaction_final_scales'),('both','initial_scales','remove_interaction_initial_scales')]
 rng=np.random.default_rng(20261109);pairs=[]
 for mode in training['modes']:
  for left,right,label in contrasts:
   for metric in ['joint','fine','raw20','nll']:
    d=np.array([np.mean([r[metric] for r in rows if r['mode']==mode and r['condition']==left and r['scene']==s])-np.mean([r[metric] for r in rows if r['mode']==mode and r['condition']==right and r['scene']==s]) for s in ref['scene_ids']]);scale=1 if metric=='nll' else 100;boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*scale
    pairs.append(dict(mode=mode,contrast=label,left=left,right=right,metric=metric,delta=float(d.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='paired scene, seeds averaged',multiplicity='uncorrected, secondary metrics descriptive'))
 a.out.mkdir(parents=True,exist_ok=False)
 (a.out/'paired.json').write_text(json.dumps(pairs,indent=2));(a.out/'validation.json').write_text(json.dumps(dict(rows=len(rows),matched_queries_per_mode_condition=1152,normal_replays=3456,initial_replays=3456,replay_mismatches=0,protocol_sha256=sha(a.run/'protocol.json'),predictions_sha256=sha(a.run/'predictions.json'),conditions=setup['conditions'],test_accessed=False),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
