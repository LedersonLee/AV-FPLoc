"""Wait for the specific live producer, validate complete inputs, record execution SHA."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path('/home/jeongeonlee/track1/experiments/material_observability_20261111')
CODE=Path(__file__).resolve().parent
while subprocess.run(['docker','inspect','--format','{{.State.Running}}','material-4a-controlled-20261111'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True).stdout.strip()=='true':time.sleep(2)
assert len(json.loads((ROOT/'controlled/manifest.json').read_text()))==1980,'Renderer incomplete: preserve all partial results'
records=[]
for name in ['analyze.py','results.py','auxiliary.py']:
 rec=dict(script=name,sha=hashlib.sha256((CODE/name).read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),start=time.time())
 with (ROOT/(name+'.log')).open('w') as log:r=subprocess.run(['python3',str(CODE/name),'--root',str(ROOT)],stdout=log,stderr=subprocess.STDOUT)
 rec.update(exit_code=r.returncode,seconds=time.time()-rec['start']);records.append(rec);(ROOT/'analysis_execution.json').write_text(json.dumps(records,indent=2));print(rec,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
