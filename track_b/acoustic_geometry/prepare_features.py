import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .features import features
from track_b.data.geometry import canonical_rir
from .prepare_targets import read,sha


def main():
    p=argparse.ArgumentParser()
    for k in ('root','targets','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True)
    p.add_argument('--parts',nargs='+',default=['train','val'],choices=['train','val','development_test'])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    assert (a.targets/'COMPLETE').exists()
    protocol=read(a.targets/'protocol.json')
    (a.out/'protocol.json').write_text(json.dumps(dict(code_sha=a.code_sha,targets_protocol_sha256=sha(a.targets/'protocol.json'),parts=a.parts,
        spatial='48kHz first16384 samples shared RMS, FFT/window256 hop128,6logmag+12IPD, average pool64x128',
        echo='48kHz first60ms, FFT512/window64/hop16,6logmag, average pool64x128; RIR, no chirp'),indent=2))
    manifest=[]
    for part in a.parts:
      for scene in protocol['scenes'][part]:
        start=time.perf_counter();refs=read(a.targets/f'{scene}_queries.json')
        echoed=[];spatial=[];checks=[]
        for q in refs:
            path=a.root/q['rir_path'];side=read(path.with_suffix('.json'))
            assert side['role']=='Observed' and side['geometry_kind']=='furnished_native_scene'
            assert side['acoustics']['sample_rate_hz']==48000
            assert np.allclose(q['centre_xyz_m'],side['centre_xyz_m'])
            assert np.allclose(side['source_xyz_m'],side['centre_xyz_m'])
            assert side['base_acoustic_yaw_deg']==q['base_acoustic_yaw_deg']
            digest=sha(path);assert digest==side['rir_sha256']
            raw=np.load(path);body=canonical_rir(raw,q['world_yaw_deg'],q['base_acoustic_yaw_deg'])
            e,s=features(torch.from_numpy(body.copy())[None]);echoed.append(e[0].half());spatial.append(s[0].half())
            checks.append(dict(query_id=q['query_id'],rir_sha256=digest,sidecar_sha256=sha(path.with_suffix('.json'))))
        target=np.load(a.targets/f'{scene}.npz')
        package=dict(scene=scene,split=part,echo=torch.stack(echoed),spatial=torch.stack(spatial),
                     depth=torch.from_numpy(target['depth']),mask=torch.from_numpy(target['mask']),
                     poses=torch.from_numpy(target['poses']),query_ids=[q['query_id'] for q in refs])
        torch.save(package,a.out/f'{scene}.pt')
        (a.out/f'{scene}_inputs.json').write_text(json.dumps(checks,indent=2))
        row=dict(scene=scene,split=part,queries=len(refs),sha256=sha(a.out/f'{scene}.pt'),seconds=time.perf_counter()-start)
        manifest.append(row);(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(row),flush=True)
    (a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()
