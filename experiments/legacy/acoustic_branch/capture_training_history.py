"""Archive only epoch metric JSON lines from the explicitly owned pilot log."""
import json
import subprocess
from pathlib import Path
from .contracts import atomic_json


def main():
    pilot=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD/AcousticContrastive/pilots/scene_subset_v1')
    run=json.loads((pilot/'active_acoustic_run.json').read_text())['path'];launch=json.loads((pilot/run/'launch.json').read_text())
    inspected=json.loads(subprocess.check_output(['docker','inspect',launch['name']],text=True))[0]
    if inspected['Id']!=launch['container_id']:raise RuntimeError('Training container ownership mismatch')
    result=subprocess.run(['docker','logs',launch['name']],capture_output=True,text=True,check=True)
    history=[]
    for line in (result.stdout+'\n'+result.stderr).splitlines():
        try:value=json.loads(line)
        except (ValueError,TypeError):continue
        if isinstance(value,dict) and 'epoch' in value and 'train/loss' in value and 'val/loss' in value:history.append(value)
    history.sort(key=lambda r:r['epoch'])
    atomic_json(pilot/run/'history.json',history)
    print(json.dumps(dict(epochs=len(history),path=str(pilot/run/'history.json'))))


if __name__=='__main__':main()
