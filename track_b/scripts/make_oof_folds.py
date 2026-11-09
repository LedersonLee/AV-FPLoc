"""Freeze deterministic two-fold training partitions without using observations/GT."""
import argparse,hashlib,json
from pathlib import Path
from track_b.data.folds import validate_training_fold

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);raw=(a.cache/'protocol.json').read_bytes();base=json.loads(raw);digest=hashlib.sha256(raw).hexdigest();groups={}
 for sid in base['scenes']['train']:groups.setdefault(sid.split('_')[0],[]).append(sid)
 bins=[[],[]]
 for group in sorted(groups,key=lambda k:hashlib.sha256(k.encode()).hexdigest()):bins[min(range(2),key=lambda k:len(bins[k]))].extend(groups[group])
 for k in range(2):
  fold=dict(role='out_of_fold_train_scenes',source_cache_protocol_sha256=digest,train=[s for s in base['scenes']['train'] if s not in bins[k]],val=[s for s in base['scenes']['train'] if s in bins[k]],fold_id=k,selection='SHA256 source-family order, greedily balance scene counts; no outcomes',restriction='Every locally trained scorer component must be fit on this fold Train only; final-step only; original Val untouched by fold construction',exposure_limit='Upstream foundation pretraining not independently excluded; inherited development method selection remains')
  validate_training_fold(base['scenes'],fold,digest);(a.out/f'fold_{k}.json').write_text(json.dumps(fold,indent=2));print(json.dumps(fold))
if __name__=='__main__':main()
