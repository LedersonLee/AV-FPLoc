"""Independent pose/metric/reference-rule reproduction; never fits a scalar."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import atomic_json,sha
from .data import read
from .colleague_mode_comparison import REL,METHODS,KEYS,metrics
from .colleague_mode_fusion import choose


def verify(root,report,reference,source):
    out=root/REL;summary=read(report/'summary.json');rows=read(report/'records.json')
    selection=read(out/'selection_frozen.json');contract=read(out/'contract.json')
    baseline=read(source/'records.json');lookup={(r['scene_id'],r['sample_index']):r for r in baseline}
    if sha(out/'selection_frozen.json')!=summary['selection_sha256']:
        raise ValueError('Frozen val selection changed')
    if selection['test_accessed'] or not selection['selected_before_test_phase']:
        raise ValueError('Selection did not precede test phase')
    if selection['selection_split']!='val' or selection['selected']['queries']!=1704:
        raise ValueError('Unexpected selection scope')
    if len(rows)!=1472 or len(lookup)!=1472 or len({(r['scene_id'],r['sample_index']) for r in rows})!=1472:
        raise ValueError('Wrong test query coverage')
    if len({r['scene_id'] for r in rows})!=3:raise ValueError('Wrong test scene coverage')
    for name,digest in contract['reference'].items():
        if sha(reference/name)!=digest:raise ValueError('Reference source changed')
    counts=Counter()
    for row in rows:
        truth=np.array(row['gt_pose_xzyaw']);old=lookup[(row['scene_id'],row['sample_index'])]
        np.testing.assert_array_equal(truth,old['gt_pose_xzyaw'])
        for method in METHODS:
            m=row['methods'][method];pose=m['pose_xzyaw']
            distance=float(np.linalg.norm(np.array(pose[:2])-truth[:2]))
            np.testing.assert_allclose(m['xy_error_m'],distance,atol=1e-10,rtol=0)
            if pose[2] is None:
                if method!='colleague_acoustic_xy' or m['yaw_error_deg'] is not None:
                    raise ValueError('Unknown acoustic heading was silently invented')
            else:
                angle=abs((pose[2]-truth[2]+180)%360-180)
                np.testing.assert_allclose(m['yaw_error_deg'],angle,atol=1e-10,rtol=0)
        for method,previous in [('vision_only','vision_only'),('ours_acoustic','acoustic_only'),
                                ('ours_equal_fusion','equal_weight_fusion')]:
            for key in ('pose_xzyaw','xy_error_m','yaw_error_deg'):
                np.testing.assert_allclose(row['methods'][method][key],old['methods'][previous][key],atol=1e-10,rtol=0)
        proposed=row['gate']['candidates']
        ev=dict(cells=np.array([c['cell'] for c in proposed]),visual=np.array([c['visual'] for c in proposed]),
                audio=np.array([c['audio'] for c in proposed]),yaw=np.array([round(c['pose_xzyaw'][2]/10) for c in proposed]))
        decision=choose(ev,selection['selected']['scalars'],reference)
        if decision['index']!=row['gate']['selected_mode']:raise ValueError('Selected mode differs from reference')
        np.testing.assert_allclose(decision['gamma'],row['gate']['gamma'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(decision['mode_scores'],[c['score'] for c in proposed],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(row['methods']['colleague_mode_gate']['pose_xzyaw'],proposed[decision['index']]['pose_xzyaw'])
        covered=any(np.linalg.norm(np.array(c['pose_xzyaw'][:2])-truth[:2])<=1 for c in proposed)
        if covered!=row['gate']['proposal_oracle_1m']:raise ValueError('Proposal coverage differs')
        ok=row['methods']['colleague_mode_gate']['xy_error_m']<=1
        failure='success' if ok else 'verifier_miss' if covered else 'proposal_miss'
        if failure!=row['gate']['failure']:raise ValueError('Failure category differs')
        counts[failure]+=1
        # Three dense baseline score views are copied byte-for-byte, not redrawn
        # from rounded numbers or rescaled to create a different impression.
        new=np.fromfile(report/row['heatmap'],dtype=np.uint8).reshape(4,-1)
        oldheat=np.fromfile(source/old['heatmap'],dtype=np.uint8).reshape(4,-1)
        np.testing.assert_array_equal(new[[0,2,3]],oldheat[:3])
    for method in METHODS:
        m=metrics(rows,method)
        for key in KEYS:
            old=summary['micro'][method][key]
            if m[key] is None:
                if old is not None:raise ValueError('N/A metric differs')
            else:np.testing.assert_allclose(m[key],old,atol=1e-12,rtol=0)
    for scene,methods in summary['per_scene'].items():
        subset=[r for r in rows if r['scene_id']==scene]
        for method in METHODS:
            m=metrics(subset,method)
            for key in KEYS:
                if m[key] is None:assert methods[method][key] is None
                else:np.testing.assert_allclose(m[key],methods[method][key],atol=1e-12,rtol=0)
    assert counts['proposal_miss']==summary['proposal_diagnostics']['proposal_misses']
    assert counts['verifier_miss']==summary['proposal_diagnostics']['verifier_misses']
    result=dict(status='verified',queries=1472,scenes=3,methods=5,all_query_poses_and_errors_recomputed=True,
        all_metrics_reproduced=True,all_reference_gate_decisions_reproduced=True,
        all_three_baseline_heatmap_layers_identical=True,heading_NA_preserved=True,
        failures=dict(counts),selection_sha256=sha(out/'selection_frozen.json'),
        report_records_sha256=sha(report/'records.json'),report_summary_sha256=sha(report/'summary.json'),test_fitting=False)
    atomic_json(out/'independent_verification.json',result);atomic_json(report/'independent_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','report','reference','source'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(verify(a.root,a.report,a.reference,a.source))
