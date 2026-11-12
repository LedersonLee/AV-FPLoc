"""Event-driven seed42 A3 queue. No polling, retries, or interrupted-run restart."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from validate_expert_gate import validate, sha


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, **kw)


def main():
    p = argparse.ArgumentParser()
    for key in ['workspace', 'study', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--await-container', required=True)
    a = p.parse_args()
    source = a.workspace/'AV-FPLoc/scripts/research/track_a/avloc_20261111'
    control = a.study/'runs/A2_disabled_seed42'
    protocol = json.loads((control/'protocol.json').read_text())
    assert protocol['arm'] == 'A2_disabled' and protocol['seed'] == 42
    gate_path = a.study/'gates/experts_seed42_v1.json'
    gate = json.loads(gate_path.read_text())
    assert validate(a.study, 42) == gate, 'Expert evidence changed'
    assert not (a.study/'runs/A3_seed42').exists(), 'Will not overwrite/restart existing A3'
    state = json.loads(run(['docker','inspect',a.await_container],capture_output=True).stdout)[0]
    assert state['State']['Running'], 'Require a live predecessor on initial scheduling'
    assert '/study/runs/A2_disabled_seed42' in state['Config']['Cmd']
    image = 'sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318'
    assert state['Image'] == image
    a.out.mkdir(parents=True, exist_ok=False)
    snapshot = a.out/'source'; snapshot.mkdir()
    for path in source.glob('*.py'):
        shutil.copyfile(path, snapshot/path.name)
    hashes = {p.name: sha(p) for p in snapshot.glob('*.py')}
    for name, digest in protocol['source_sha256'].items():
        assert hashes[name] == digest
    gate_hash = sha(gate_path)
    cmd = ['docker','run','--name','avloc-a3-seed42-priority-20261112',
           '--gpus','all','--network','none','--cpus','8','--memory','20g','--shm-size','2g',
           '--user',f'{os.getuid()}:{os.getgid()}',
           '-e','PYTHONDONTWRITEBYTECODE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8',
           '-e','PYTORCH_KERNEL_CACHE_PATH=/tmp',
           '-v',f'{snapshot}:/source:ro',
           '-v',f'{a.workspace}/experiments/avloc_20261111/weights:/weights:ro',
           '-v',f'{a.study}:/study:rw', image, 'python','/source/train.py',
           '--data','/study/data_v1','--noise','/study/esc50_v1',
           '--weights','/weights/resnet50-11ad3fa6.pth','--out','/study/runs/A3_seed42',
           '--arm','A3','--seed','42','--batch','8','--workers','4',
           '--code-sha',protocol['code_commit'],'--expert-gate','/study/gates/experts_seed42_v1.json']
    plan = dict(predecessor_id=state['Id'], command=cmd, source_sha256=hashes,
                gate_sha256=gate_hash, training_source_commit=protocol['code_commit'],
                policy='wait exact container; require exit0 and training_complete; revalidate experts; no retry/overwrite; seed42 only', test_accessed=False)
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('WAIT LIVE',state['Id'],flush=True)
    result = run(['docker','wait',state['Id']],capture_output=True)
    assert result.stdout.strip() == '0', result.stdout
    assert json.loads((control/'complete.json').read_text())['status'] == 'training_complete'
    for name, digest in hashes.items():
        assert sha(snapshot/name) == digest
    assert sha(gate_path) == gate_hash and validate(a.study,42) == gate
    assert not (a.study/'runs/A3_seed42').exists()
    print('START A3 seed42',flush=True)
    with (a.out/'A3_seed42.log').open('x') as f:
        run(cmd,stdout=f,stderr=subprocess.STDOUT)
    assert json.loads((a.study/'runs/A3_seed42/complete.json').read_text())['status'] == 'training_complete'
    (a.out/'complete.json').write_text(json.dumps(dict(status='A3_training_complete',benchmark_complete=False))+'\n')


if __name__ == '__main__':
    main()
