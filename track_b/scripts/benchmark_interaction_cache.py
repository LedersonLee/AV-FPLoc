"""Separate learned map-cache setup and full-field tensor-input query latency."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from track_b.models.interaction import InteractionMatcher

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2);protocol=json.loads((a.cache/'protocol.json').read_text());rows=[]
 for mode in ('attention','concat','separable'):
  m=InteractionMatcher(mode).cuda().eval();m.load_state_dict(torch.load(a.run/f'{mode}_42.pt',weights_only=True));checkpoint_bytes=(a.run/f'{mode}_42.pt').stat().st_size
  for sid in protocol['scenes']['val']:
   s=torch.load(a.cache/f'{sid}.pt',weights_only=True,map_location='cpu');rgb=s['rgb'][:8].cuda().float()/255;spec=s['spectrum'][:8].cuda().float();fp=s['floorplan'].cuda();p=s['candidates'].cuda();cal={'origin':s['origin'].cuda(),'spacing':s['spacing'],'hfov_deg':s['hfov_deg']};torch.cuda.reset_peak_memory_stats();setup=[]
   with torch.no_grad():
    for _ in range(3):
     torch.cuda.synchronize();start=time.perf_counter();cache=m.prepare_map(fp,cal,p);torch.cuda.synchronize();setup.append(time.perf_counter()-start)
    expected=m(rgb[:1],spec[:1],fp,cal,p);actual=m.forward_cached(rgb[:1],spec[:1],cache);difference=float((expected-actual).abs().max());torch.testing.assert_close(actual,expected,atol=3e-5,rtol=1e-5)
    same_map=bool(actual.argmax()==expected.argmax());times=[]
    for i in range(22):
     q=i%8;torch.cuda.synchronize();start=time.perf_counter();field=m.forward_cached(rgb[q:q+1],spec[q:q+1],cache);torch.cuda.synchronize()
     if i>=2:times.append((time.perf_counter()-start)*1000)
   row=dict(mode=mode,scene=sid,candidates=len(p),checkpoint_bytes=checkpoint_bytes,candidate_input_bytes=p.numel()*p.element_size(),learned_map_cache_bytes=sum(v.numel()*v.element_size() for v in cache.values() if isinstance(v,torch.Tensor)),map_setup_seconds=setup,warm_query_ms_mean=float(np.mean(times)),warm_query_ms_median=float(np.median(times)),warm_query_ms_p95=float(np.quantile(times,.95)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),cache_uncached_max_score_difference=difference,cache_uncached_argmax_equal=same_map,scope='CUDA synced;20 timed single queries after2 warmups; eight preprocessed observations cyclically reused; excludes disk decode/RIR normalization/STFT and geometric ray setup; no end-to-end sensor latency claim')
   rows.append(row);a.out.write_text(json.dumps(dict(status='running',rows=rows),indent=2));print(json.dumps(row),flush=True);del s,rgb,spec,fp,p,cal,cache,expected,actual,field;torch.cuda.empty_cache()
  del m;torch.cuda.empty_cache()
 a.out.write_text(json.dumps(dict(status='complete',rows=rows),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
