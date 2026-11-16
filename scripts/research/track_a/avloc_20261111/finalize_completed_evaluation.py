"""Post-evaluation verification queue. No training, overwrite, or publication."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from predict_hash import sha


def main():
    p = argparse.ArgumentParser()
    for k in ['workspace', 'study', 'evaluation-queue', 'out']:
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    snapshot = a.out/'source'
    shutil.copytree(Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {q.name:sha(q) for q in snapshot.glob('*.py')}
    image = 'sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318'
    stages = []
    native = a.study/'evaluation/A3_seed42_10cm/summary.json'
    done = a.evaluation_queue/'complete.json'
    base = '/study/comparisons/A3_final_20261116'
    stages.append(('integrity', native, ['validate_expert_gate.py','--study','/study','--seed','42','--arm','A3','--out',base+'/integrity.json']))
    stages.append(('metrics', native, ['requested_metric_table.py','--study','/study','--out',base+'/metrics','--include-arm','A2_disabled','--include-arm','A3']))
    for grid in [20, 10]:
        stages.append((f'paired{grid}', done, ['compare_router.py','--study','/study','--out',base+f'/paired{grid}','--grid-cm',str(grid),'--include-uniform']))
        stages.append((f'calibration_review{grid}', done, ['summarize_ray_calibration.py','--baseline',f'/study/evaluation/A3_seed42_{grid}cm','--calibration',f'/study/calibration/A3_seed42_{grid}cm','--out',base+f'/calibration_review{grid}']))
    stages.append(('figures', done, ['plot_matched_fields.py','--workspace',str(a.workspace),'--study','/study','--out',base+'/figures','--include-arm','A2_disabled','--include-arm','A3']))
    (a.out/'plan.json').write_text(json.dumps(dict(image=image,source_sha256=hashes,stages=[(n,str(g),args) for n,g,args in stages],policy='Wait for verified completion; stop on errors; preserve all existing outputs; no publication'),indent=2)+'\n')
    for name, gate, args in stages:
        print('WAIT', name, str(gate), flush=True)
        while not gate.exists():
            time.sleep(60)
        receipt = json.loads(gate.read_text())
        if gate == done:
            assert receipt['status'] == 'completed_expert_evaluation_complete'
        else:
            assert receipt['complete'] and receipt['queries'] == 9965
        for filename, digest in hashes.items():
            assert sha(snapshot/filename) == digest
        output = args[args.index('--out')+1]
        assert not (a.study/Path(output).relative_to('/study')).exists()
        cmd = ['docker','run','--rm','--network','none','--cpus','2','--memory','8g','--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-v',f'{snapshot}:/source:ro','-v',f'{a.study}:/study:rw','-v',f'{a.workspace}:{a.workspace}:ro',image,'python','/source/'+args[0],*args[1:]]
        with (a.out/(name+'.log')).open('x') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        (a.out/(name+'_complete.json')).write_text(json.dumps(dict(complete=True,gate_sha256=sha(gate),output=output))+'\n')
        print('COMPLETE',name,flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(complete=True,publication_pending=True,visual_review_pending=True,test_accessed=False))+'\n')


if __name__ == '__main__':
    main()
