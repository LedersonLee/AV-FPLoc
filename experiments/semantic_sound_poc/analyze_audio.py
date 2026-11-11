"""Analyze recorded channel timing; generate synthetic broadband convolution only."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile

def onset(arr):
 return np.array([int(np.flatnonzero(abs(x)>np.max(abs(x))*1e-3)[0]) if np.any(x) else -1 for x in arr])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);o=ap.parse_args().out;c=json.loads((o/'config.json').read_text());render=json.loads((o/'acoustic_render.json').read_text());rows=[]
 for rec in render['records']:
  arr=np.load(o/'rirs'/(rec['name']+'.npy'));m=np.array(rec['mic_positions']);expected=np.linalg.norm(m-rec['source'],axis=1)/c['sound_speed']*c['sample_rate'];t=onset(arr);error=(t-t[0])-(expected-expected[0]);rows.append(dict(name=rec['name'],pair=rec['pair'],heading=rec['heading'],seed=rec['seed'],furnished=rec['furnished'],onset_samples=t.tolist(),expected_samples=expected.tolist(),tdoa_error_samples=error.tolist(),max_abs_tdoa_error_samples=float(max(abs(error[1:]))),absolute_arrival_bias_samples=float(np.median(t-expected)),pass_1sample=bool(max(abs(error[1:]))<=1)))
 groups=[]
 for pair in range(3):
  for heading in [0,90]:
   rr=[r for r in rows if r['furnished'] and r['pair']==pair and r['heading']==heading];t=np.array([r['onset_samples'] for r in rr]);groups.append(dict(pair=pair,heading=heading,repeat_onset_std_samples=t.std(0).tolist(),max_abs_tdoa_error_samples=max(r['max_abs_tdoa_error_samples'] for r in rr),all_pass_1sample=all(r['pass_1sample'] for r in rr)))
 direct=json.loads((o/'direct_probe.json').read_text());drows=[]
 for r in direct['records']:
  arr=np.load(o/'direct_probe'/(r['name']+'.npy'));r['onset_samples']=onset(arr).tolist();drows.append(r)
 # Waveform difference, late-energy change vs Monte Carlo repeat; source/receiver identical.
 stats={}
 for role in ['furnished','stage']:
  vals=[]
  for seed in [42,43,44]:
   arr=np.load(o/'rirs'/f'{role}_p0_h0_s{seed}.npy');vals.append(float(np.sum(arr[:,250:]**2)))
  stats[role]=dict(late_energy_each=vals,mean=float(np.mean(vals)),std=float(np.std(vals)))
 arr=np.load(o/'rirs/furnished_p0_h0_s42.npy');rng=np.random.default_rng(42);signal=rng.normal(size=4800);signal*=np.hanning(len(signal));convolved=np.stack([fftconvolve(signal,ch) for ch in arr],axis=1);scale=.9/max(abs(convolved.ravel()));wavfile.write(o/'broadband_received_6ch.wav',48000,(convolved*scale).astype(np.float32));wavfile.write(o/'broadband_source.wav',48000,signal.astype(np.float32))
 result=dict(rows=rows,groups=groups,direct_probe=drows,furniture_intervention=stats,quantization_us=1e6/48000,timing_threshold_samples=1,valid_direct_configurations=sum(g['all_pass_1sample'] for g in groups),total_configurations=6,physical_array_status='PASS' if all(g['all_pass_1sample'] for g in groups) else 'FAIL: partial direct-arrival validation; retain failed configuration',absolute_time_warning='Compare point-center distance/c separately; systematic early arrival is not corrected or hidden. Source radius/default backend handling unverified in this PoC.',waveform='generated Gaussian broadband, seed42, 100ms Hann window; not a licensed household event recording',convolution_scale=scale)
 (o/'audio_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(groups,indent=2));print(stats)
if __name__=='__main__':main()
