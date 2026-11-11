"""Bounded documented-C-API diagnostic; original RIRs are read-only."""
import argparse,json,subprocess,hashlib,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--executable',type=Path,required=True);p.add_argument('--modes',default='baseline,radius0,indirect_off,direct_off,temporal_on,simultaneous,stationary_off,stationary_on');a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);rows=[]
 for d in [2.,6.]:
  for mode in a.modes.split(','):
   for context in range(3):
    prefix=a.out/f'd{d}_{mode}_c{context}';repeat=5 if mode.startswith('stationary') else 1;start=time.perf_counter()
    with Path(str(prefix)+'.log').open('w') as f:subprocess.run([str(a.executable),str(prefix),mode,str(d),str(repeat)],stdout=f,stderr=subprocess.STDOUT,check=True)
    r=json.loads(Path(str(prefix)+'.json').read_text());r['context']=context;r['receipt_sha256']=hashlib.sha256(Path(str(prefix)+'.json').read_bytes()).hexdigest();r['file_prefix']=prefix.name;r['wall_seconds']=time.perf_counter()-start;r['files']=[dict(name=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in sorted(a.out.glob(prefix.name+'*.f32'))];rows.append(r);(a.out/'manifest.json').write_text(json.dumps(rows,indent=2));print(mode,d,context,round(r['seconds'],3),flush=True)
if __name__=='__main__':main()
