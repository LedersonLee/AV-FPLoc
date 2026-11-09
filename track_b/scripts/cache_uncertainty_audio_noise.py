"""Offline body-RIR synthetic noise, with exact clean-spectrum replay.

This is not ESS recording noise. Schedule angles are offline assembly metadata;
only the transformed spectrum is passed to the localization model.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from track_b.data.geometry import canonical_rir
from track_b.models.structure import rir_spectrum

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for k in ['root','cache','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
    data=read(a.cache/'protocol.json');assert len(data['scenes']['val'])==19 and data['queries_per_scene']==64
    assert data['sample_rate_hz']==48000 and data['rir_samples']==16384
    a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(code_sha=a.code_sha,data=data,role='offline observed body-RIR corruption; no reference RIR',snr_db=20,seed=20261109,noise='Gaussian array rescaled to exact joint6ch RMS ratio0.1 before standard noisy-waveform RMS normalization/STFT/IPD',rng='NumPy default_rng seeded by first8bytes little-endian SHA256 of 20261109:scene:query',alignment='existing schedule canonical_rir; world/base yaw used offline only, never model inputs',limitation='synthetic RIR perturbation, not acoustic recording/ESS deconvolution noise',test_accessed=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest=[];total=0
    for sid in data['scenes']['val']:
        original=torch.load(a.cache/f'{sid}.pt',weights_only=True,map_location='cpu');inputs=read(a.cache/f'{sid}_inputs.json')
        schedule=read(a.root/f'RIR/manifests/positive_pairs_v1/schedules/val/{sid}.json');mapping={q['sample_index']:q for q in schedule['queries']}
        assert original['query_ids']==[x['query_id'] for x in inputs];specs=[];snrs=[]
        for i,item in enumerate(inputs):
            q=mapping[item['query_id']];assert q['observed_rir_path']==item['rir_path']
            path=a.root/item['rir_path'];assert '/RIR/Observed/' in str(path) and sha(path)==item['rir_sha256']
            assert read(path.with_suffix('.json'))['role']=='Observed'
            body=canonical_rir(np.load(path),q['world_yaw_deg'],q['base_acoustic_yaw_deg']);wave=np.zeros((6,16384),np.float32);n=min(body.shape[-1],16384);wave[:,:n]=body[:,:n]
            rms=float(np.sqrt(np.mean(wave**2)));assert np.isfinite(wave).all() and rms>1e-8
            clean=rir_spectrum(torch.from_numpy(wave/rms)[None])[0].half()
            assert torch.equal(clean,original['spectrum'][i]), f'Clean preprocessing replay mismatch {sid}/{i}'
            seed=int.from_bytes(hashlib.sha256(f"20261109:{sid}:{item['query_id']}".encode()).digest()[:8],'little')
            noise=np.random.default_rng(seed).standard_normal(wave.shape).astype(np.float32)
            noise*=rms*.1/float(np.sqrt(np.mean(noise**2)))
            snr=20*math.log10(rms/float(np.sqrt(np.mean(noise**2))));assert abs(snr-20)<1e-4;snrs.append(snr)
            noisy=wave+noise;noisy/=max(float(np.sqrt(np.mean(noisy**2))),1e-8)
            spec=rir_spectrum(torch.from_numpy(noisy)[None])[0].half();assert torch.isfinite(spec).all();specs.append(spec);total+=1
        path=a.out/f'{sid}.pt';torch.save(dict(scene=sid,query_ids=original['query_ids'],spectrum=torch.stack(specs)),path)
        manifest.append(dict(scene=sid,queries=len(specs),sha256=sha(path),input_manifest_sha256=sha(a.cache/f'{sid}_inputs.json'),clean_replay_mismatches=0,snr_min=min(snrs),snr_max=max(snrs)))
        (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',sid,flush=True)
    assert total==1216
    (a.out/'validation.json').write_text(json.dumps(dict(queries=total,scenes=19,clean_spectrum_bitwise_replay_mismatches=0,snr_db=20,test_accessed=False),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
