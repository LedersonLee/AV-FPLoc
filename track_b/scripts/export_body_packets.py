"""Offline dataset assembly: materialize body-frame observations, not inference."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from track_b.data.geometry import canonical_rir
import hashlib
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['root','cache','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);protocol=read(a.cache/'protocol.json');rows=[]
 for sid in protocol['scenes']['val']:
  source=read(a.cache/f'{sid}_inputs.json')[0]
  schedule=read(a.root/f'RIR/manifests/positive_pairs_v1/schedules/val/{sid}.json')
  q=next(q for q in schedule['queries'] if q['sample_index']==source['query_id'])
  assert q['observed_rir_path']==source['rir_path'] and q['image_path']==source['rgb_path']
  wave=a.root/source['rir_path'];image=a.root/source['rgb_path'];assert sha(wave)==source['rir_sha256'] and sha(image)==source['rgb_sha256']
  side=read(wave.with_suffix('.json'));assert side['role']=='Observed'
  body=canonical_rir(np.load(wave),q['world_yaw_deg'],q['base_acoustic_yaw_deg'])
  outwave=a.out/f'{sid}_body_rir.npy';outimage=a.out/f'{sid}_rgb.png';np.save(outwave,body);shutil.copyfile(image,outimage)
  rows.append(dict(scene=sid,query=source['query_id'],body_rir=outwave.name,rgb=outimage.name,body_rir_sha256=sha(outwave),rgb_sha256=sha(outimage),observed_source_sha256=sha(wave),sample_rate_hz=48000))
 (a.out/'manifest.json').write_text(json.dumps(rows,indent=2))
 (a.out/'protocol.json').write_text(json.dumps(dict(selection='first previously frozen query in each of six Val scenes; no outcome selection',role='offline dataset assembly, not GT-free inference',alignment='existing schedule world/base yaw used only to materialize ring-symmetric body-frame waveform; inference never receives these angles',limitation='does not demonstrate calibration without metadata on an unaligned physical sensor',cache_protocol_sha256=sha(a.cache/'protocol.json'),test_accessed=False),indent=2));print('COMPLETE')
if __name__=='__main__':main()
