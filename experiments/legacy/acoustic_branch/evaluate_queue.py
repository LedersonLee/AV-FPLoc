"""Local evaluation waits for BOTH training and independently generated bank."""
import argparse
import fcntl
from pathlib import Path
import time
import traceback

from .contracts import atomic_json
from .data import read


def run(args):
    pilot=args.root/'AcousticContrastive/pilots/scene_subset_v1';evaluation=pilot/'evaluation'
    args.report.mkdir(parents=True,exist_ok=True)
    lock=(args.report/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        while True:
            run=read(pilot/'active_acoustic_run.json')['path'] if (pilot/'active_acoustic_run.json').exists() else 'runs/acoustic_v1'
            training=read(pilot/run/'queue_status.json') if (pilot/run/'queue_status.json').exists() else {}
            test=read(evaluation/'status.json') if (evaluation/'status.json').exists() else {}
            if training.get('status')=='failed' or test.get('status')=='failed':
                raise RuntimeError('Upstream training or test bank failed; no results fabricated')
            if training.get('status')=='complete' and (evaluation/'ready.json').exists():break
            progress_path=evaluation/'render/status.json'
            progress=read(progress_path) if progress_path.exists() else {}
            progress={k:v for k,v in progress.items() if k in ('completed_tasks','tasks','failed_tasks','records_written','records_verified_existing','total_records','updated_unix')}
            atomic_json(args.report/'status.json',dict(status='waiting',training=training,
                test_bank=test,render_progress=progress,updated_unix=time.time(),results_available=False))
            time.sleep(30)
        atomic_json(args.report/'status.json',dict(status='running',stage='four_way_evaluation',updated_unix=time.time()))
        from .pilot_evaluate import run as evaluate
        evaluate(args)
        atomic_json(args.report/'status.json',dict(status='complete',stage='report_ready',
            updated_unix=time.time(),results_available=True))
    except Exception:
        atomic_json(args.report/'status.json',dict(status='failed',error=traceback.format_exc(),
            updated_unix=time.time(),results_available=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vision-hssd',type=Path,required=True);p.add_argument('--vision-eval',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=256)
    run(p.parse_args())
