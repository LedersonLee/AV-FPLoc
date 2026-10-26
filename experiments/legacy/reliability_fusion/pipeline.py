"""Persistent local queue: verified inputs -> frozen scores -> train -> test HTML."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import traceback

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from .prepare import REL


def run(args):
    out=args.root/REL;args.report.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        value=dict(status='running',stage=stage,updated_unix=time.time(),**extra)
        atomic_json(out/'pipeline_status.json',value);atomic_json(args.report/'status.json',value)
    def execute(module,*arguments):
        status(module);subprocess.run([sys.executable,'-u','-m',module,*map(str,arguments)],check=True)
    try:
        while True:
            pre=read(out/'preprocess_status.json') if (out/'preprocess_status.json').exists() else {}
            if pre.get('status')=='failed':raise RuntimeError('Node2 preprocessing failed: '+str(pre.get('error')))
            vision=read(out/'vision_ready.json') if (out/'vision_ready.json').exists() else {}
            if pre.get('status')=='complete' and vision.get('status')=='complete':break
            render=read(out/'render/status.json') if (out/'render/status.json').exists() else {}
            status('waiting_verified_preprocessing',preprocessing=pre,render=render,vision=vision)
            time.sleep(30)
        if not (out/'scores_ready.json').exists():
            execute('reliability_fusion.cache','--root',args.root,'--vision-root',args.vision_root)
        if not (out/'runs/training_complete.json').exists():execute('reliability_fusion.train','--root',args.root)
        if not (out/'test_scores_ready.json').exists():
            execute('reliability_fusion.cache','--root',args.root,'--vision-root',args.vision_root,'--test')
        execute('reliability_fusion.report','--root',args.root,'--vision-root',args.vision_root,'--report',args.report)
        # Shared copy has identical static assets and all source lineage.
        import shutil
        shared=out/'report';shared.mkdir(exist_ok=True)
        shutil.copytree(args.report,shared,dirs_exist_ok=True)
        atomic_json(out/'pipeline_status.json',dict(status='complete',stage='report_ready',report=str(args.report),updated_unix=time.time()))
    except Exception:
        error=dict(status='failed',error=traceback.format_exc(),updated_unix=time.time())
        atomic_json(out/'pipeline_status.json',error);atomic_json(args.report/'status.json',error);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','vision-root','report'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
