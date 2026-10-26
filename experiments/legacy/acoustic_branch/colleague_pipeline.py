"""Fresh yaw0 projection fit, selected checkpoint, then held-out test report."""
import argparse
import fcntl
from pathlib import Path
import shutil
import time
import traceback

from .colleague_projection import REL,cache_train_val,train
from .colleague_evaluate import evaluate
from .contracts import atomic_json


def main(args):
    out=args.root/REL;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage):atomic_json(out/'status.json',dict(status='running',stage=stage,updated_unix=time.time()))
    try:
        status('verify_train_val_features');cache_train_val(args.root,args.reference,args.workers)
        status('fit_linear_projection');train(args.root,args.wandb_mode)
        status('held_out_dense_grid_test');evaluate(args)
        shutil.copytree(args.report,out/'report',dirs_exist_ok=True)
        atomic_json(out/'status.json',dict(status='complete',stage='report_ready',updated_unix=time.time()))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('root','reference','vision-root','vision-eval','report'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda')
    p.add_argument('--wandb-mode',default='online')
    main(p.parse_args())
