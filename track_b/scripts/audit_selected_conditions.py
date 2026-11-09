"""Audit physical conditions only in already-selected Train/Val observation sidecars."""
import argparse,collections,hashlib,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();protocol=json.loads((a.cache/'protocol.json').read_text());counts=collections.Counter();scenes=[]
 for split,sids in protocol['scenes'].items():
  assert split in ('train','val')
  for sid in sids:
   rows=json.loads((a.cache/f'{sid}_inputs.json').read_text());conditions=collections.Counter();digests=[]
   for r in rows:
    assert r['rir_path'].startswith(f'RIR/Observed/{split}/{sid}/');p=(a.root/r['rir_path']).with_suffix('.json');raw=p.read_bytes();d=json.loads(raw);assert d['role']=='Observed' and d['split']==split and d['rir_sha256']==r['rir_sha256'];key=(d['geometry_kind'],d['acoustics'].get('materials'),d['acoustics'].get('transmission'));conditions[key]+=1;counts[key]+=1;digests.append(hashlib.sha256(raw).hexdigest())
   scenes.append(dict(scene=sid,split=split,queries=len(rows),conditions=[dict(geometry=k[0],materials=k[1],transmission=k[2],count=v) for k,v in conditions.items()],ordered_sidecar_hash_digest=hashlib.sha256('\n'.join(digests).encode()).hexdigest()))
 result=dict(scope='selected22 Train/Val scenes,1408 observed sidecars only; no new Test observations',conditions=[dict(geometry=k[0],materials=k[1],transmission=k[2],count=v) for k,v in counts.items()],scenes=scenes,interpretation='Furnished-native observations are used; no paired furnished/unfurnished or material-varied query condition in this selected cache. This is not a global dataset absence claim.')
 a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result['conditions']))
if __name__=='__main__':main()
