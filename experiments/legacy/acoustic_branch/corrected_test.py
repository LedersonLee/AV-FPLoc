"""Frozen corrected test bank, post-training four-way evaluation, HTML handoff.

Node2 uses a mount alias RIR/FP/test -> FP_corrected_yup_v2/test. It cannot
see/reuse invalid old test FP arrays. No fitted parameters depend on test data.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import time
import traceback

from .contracts import atomic_json, immutable_json, sha
from .data import read, splits, safe_path
from .corrected_pipeline import REL as TRAIN_REL

REL = TRAIN_REL+'/evaluation'
RUN = TRAIN_REL+'/runs/acoustic_fresh_v1'
EVAL_RELATIVE = 'corrected_yup_v2/evaluation'
RUN_RELATIVE = 'corrected_yup_v2/runs/acoustic_fresh_v1'


def prepare(root):
    from .render_adapter import positive_tools
    config_path=root/TRAIN_REL/'config.json'
    config=read(config_path)
    original=root/'AcousticContrastive/pilots/scene_subset_v1/evaluation/bank_index.json'
    index=copy.deepcopy(read(original))
    if ({e['scene_id'] for e in index['scenes']}!=set(splits(root,config)['test'])
        or index['split_sha256']!=config['split_sha256']
        or index['subset_sha256']!=config['scene_subset_sha256']):
        raise ValueError('Frozen test bank/split mismatch')
    for entry in index['scenes']:
        scene=entry['scene_id']
        for key,digest in [('schedule_path','schedule_sha256'),('observation_schedule_path','observation_schedule_sha256')]:
            if sha(safe_path(root,entry[key]))!=entry[digest]:
                raise ValueError('Frozen test schedule changed')
        geometry=root/'RIR/Geometry'/scene/'floorplan.glb'
        contract=read(geometry.with_suffix('.json'))
        if sha(geometry)!=contract['glb_sha256']:
            raise ValueError('Test extrusion mesh changed')
        immutable_json(geometry.with_name('floorplan.y_up.stage_config.json'),
            dict(render_asset='floorplan.glb',up=[0,1,0],front=[0,0,-1],units_to_meters=1.))
        # The native render container intentionally mounts Geometry read-only.
        # Generate every furnished registry before launch, including scenes whose
        # observed RIRs were only partially available from an earlier run.
        positive_tools().furnished_registry(root,root/'source/hssd-hab',scene)
    index.update(version='corrected_yup_test_bank_v2',config_sha256=sha(config_path),
                 original_bank_sha256=sha(original),frame_version='corrected_yup_v2')
    immutable_json(root/REL/'bank_index.json',index)
    immutable_json(root/REL/'protocol.json',dict(config_sha256=sha(config_path),
        bank_sha256=sha(root/REL/'bank_index.json'),checkpoint='best.pt selected by VAL loss only',
        methods=['vision_only','acoustic_only','equal_weight_fusion','visual_topk_acoustic_rerank'],
        primary_k=50,diagnostic_ks=[5,10,20],grid_spacing_m=.2,yaw_step_deg=10,
        test_scenes=[e['scene_id'] for e in index['scenes']],queries=sum(e['queries'] for e in index['scenes']),
        raw_prefix='RIR/FP_corrected_yup_v2/test',no_test_fitting=True,
        visual_checkpoint='UnLoc/evaluation_checkpoints/epoch020/model.ckpt'))
    print(dict(test_scenes=len(index['scenes']),queries=sum(e['queries'] for e in index['scenes']),
               fp_arrays=index['fp_arrays'],observed_arrays=index['observed_arrays']),flush=True)


def bank(a):
    from . import test_rirs
    from .render_adapter import supervisor_tools
    # Only this new process changes the module's workspace; old reports/queues
    # and all immutable old manifests stay untouched.
    test_rirs.REL=REL
    out=a.root/REL
    lock=(out/'bank.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_path=a.root/TRAIN_REL/'config.json'
    config=read(config_path)
    def status(stage,**extra):
        atomic_json(out/'status.json',dict(status='running',stage=stage,updated_unix=time.time(),**extra))
    try:
        while True:
            upstream=read(a.root/TRAIN_REL/'status.json')
            if upstream.get('status')=='failed':
                raise RuntimeError('Corrected train preprocessing failed; cannot proceed')
            if upstream.get('status')=='complete':break
            status('waiting_node2_train_preprocessing',upstream_stage=upstream.get('stage'))
            time.sleep(30)
        index=read(out/'bank_index.json')
        if index['config_sha256']!=sha(config_path):raise ValueError('Test config changed')
        tasks=test_rirs.tasks_for(a.root,a.renderer,index)
        status('render_corrected_test_rirs',fp_arrays=index['fp_arrays'],observed_arrays=index['observed_arrays'])
        supervisor=supervisor_tools()
        supervisor.supervise_tasks(tasks,out/'render',38,expected_records=index['fp_arrays']+index['observed_arrays'])
        status('verify_and_extract_test_features')
        features={}
        for entry in index['scenes']:
            features[entry['scene_id']]=test_rirs.extract_scene(a.root,entry,config,workers=48)
        # Extraction validates every raw SHA/schedule, and every FP loaded frame;
        # receipt hashes pin the exact arrays and features used by evaluation.
        verification=dict(status='complete',bank_sha256=sha(out/'bank_index.json'),
            config_sha256=sha(config_path),frame_version='corrected_yup_v2',features=features,
            verified_fp_arrays=index['fp_arrays'],verified_observed_arrays=index['observed_arrays'])
        atomic_json(out/'raw_verification.json',verification)
        atomic_json(out/'ready.json',dict(status='complete',bank_sha256=sha(out/'bank_index.json'),
            config_sha256=sha(config_path),features=features,frame_version='corrected_yup_v2',
            raw_verification_sha256=sha(out/'raw_verification.json'),updated_unix=time.time()))
        atomic_json(out/'status.json',dict(status='complete',stage='corrected_test_bank_ready',updated_unix=time.time()))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()))
        raise


def evaluate_queue(a):
    out=a.root/REL
    a.report.mkdir(parents=True,exist_ok=True)
    lock=(a.report/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        while True:
            training=read(a.root/RUN/'queue_status.json') if (a.root/RUN/'queue_status.json').exists() else {}
            test=read(out/'status.json') if (out/'status.json').exists() else {}
            if training.get('status')=='failed' or test.get('status')=='failed':
                raise RuntimeError('Training or corrected test preprocessing failed; evaluation blocked')
            if training.get('status')=='complete' and test.get('status')=='complete' and (out/'ready.json').exists():break
            render=read(out/'render/status.json') if (out/'render/status.json').exists() else {}
            atomic_json(a.report/'status.json',dict(status='waiting',training=training,test_bank=test,
                render_progress=render,results_available=False,updated_unix=time.time()))
            time.sleep(30)
        from .pilot_evaluate import run
        a.config=a.root/TRAIN_REL/'config.json'
        a.evaluation_relative=EVAL_RELATIVE
        a.run_relative=RUN_RELATIVE
        atomic_json(a.report/'status.json',dict(status='running',stage='four_way_evaluation',results_available=False,updated_unix=time.time()))
        run(a)
        # Self-contained artifact checks before advertising report completion.
        records=read(a.report/'records.json')
        summary=read(a.report/'summary.json')
        index=read(out/'bank_index.json')
        expected=sum(e['queries'] for e in index['scenes'])
        if len(records)!=expected or summary['queries']!=expected or not summary.get('corrected_frame'):
            raise ValueError('Incomplete corrected test report')
        for name in ('index.html','metrics.csv','results.md','banks.json'):
            if not (a.report/name).is_file():raise FileNotFoundError(name)
        for record in records:
            if not (a.report/record['heatmap']).is_file():raise FileNotFoundError(record['heatmap'])
        atomic_json(a.report/'status.json',dict(status='complete',stage='report_ready',queries=expected,
            results_available=True,summary_sha256=sha(a.report/'summary.json'),updated_unix=time.time()))
    except Exception:
        atomic_json(a.report/'status.json',dict(status='failed',error=traceback.format_exc(),results_available=False,updated_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','bank','evaluate_queue'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--renderer',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--vision-hssd',type=Path)
    p.add_argument('--vision-eval',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root)
    else:globals()[a.mode](a)
