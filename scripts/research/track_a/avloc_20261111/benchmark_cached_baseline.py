"""Warm CPU timing from cached baseline features, never an end-to-end claim."""
import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
import joblib
import sklearn
import scipy
import threadpoolctl
import numpy as np
from evaluate_baseline_fields import sha
from matching import proposals


def read(p):
    return json.loads(Path(p).read_text())


def timed(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns()-start)/1e6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['workspace', 'study', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    assert a.warmups >= 1 and a.repeats >= 1
    sys.path.insert(0, str(a.workspace/'AV-FPLoc/src'))
    from avfploc.geometry.rays import angular_rays
    from avfploc.evaluation.baselines import standardized
    auditp = a.workspace/'experiments/avloc_20261111/baseline_input_audit.json'
    audit = read(auditp)
    assert audit['complete'] and audit['queries'] == 9965
    manifest = read(a.study/'data_v1/manifest.json')
    scenes = [r for r in manifest['records'] if r['split'] == 'val']
    assert len(scenes) == 19 and sum(r['queries'] for r in scenes) == 9965
    rng = np.random.default_rng(20261111)
    selected = [(r['scene'], int(rng.integers(r['queries']))) for r in scenes]
    exp = a.workspace/'experiments'
    georoot = exp/'structural_evidence_20261107_1730'
    spec = importlib.util.spec_from_file_location('frozen_geoq', georoot/'inference.py')
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    selection = read(georoot/'final_selection_seed42.json')
    cp = georoot/'GEOQ_seed42/model.joblib'
    assert sha(cp) == selection['model_sha256']
    model = joblib.load(cp)
    frozen_geo = {(r['scene'],r['query_id']):r for r in read(a.study/'baselines/geoq_replay_v1/seed42_queries.json')}
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    for sid, i in (selected[:1] if a.smoke else selected):
        rec = next(r for r in audit['records'] if r['scene'] == sid)
        folder = a.workspace/rec['folder_relative']
        for name, digest in {**rec['files'], 'receipt.json':rec['receipt_sha256']}.items():
            assert sha(folder/name) == digest
        with np.load(folder/'query.npz') as z:
            qid = int(z['sample_indices'][i])
            qmu, qb, qa = (z[k][i].astype(float) for k in ['mu','scale','observed'])
        with np.load(a.study/f'data_v1/{sid}/labels.npz') as labels:
            assert int(labels['query_ids'][i]) == qid
        xy = np.load(folder/'xy.npy')
        win = np.load(folder/'windows.npy').astype(float)
        fp = np.load(folder/'fp.npy').astype(float)
        hroot = exp/'hypothesis_discrimination_20261107_1538/cache_seed42'
        hp = next(hroot/s/(sid+'.npz') for s in ['val','extra'] if (hroot/s/(sid+'.npz')).exists())
        hr = read(hp.with_suffix('.json'))
        assert sha(hp) == hr['output_sha256']
        paths = [exp/f'astra_fusion_v2_20261107_0225/support_seed42/{s}/{sid}_adapted.npz' for s in ['val','extra_val']]+[exp/f'fusion_vs_posterior_20261107_1300/{s}_seed42/{sid}_adapted.npz' for s in ['cache','confirmation']]
        ap = next(p for p in paths if p.exists() and sha(p) == hr['adapted_sha256'])
        with np.load(ap) as z:
            assert int(z['query_ids'][i]) == qid
            bmu, bb, ba = (z[k][i].astype(float) for k in ['mu','scale','observed'])
        def visual(mu, scale):
            mu, scale = angular_rays(mu, scale)
            return -(abs(win-mu)/scale).sum(-1)/40
        funcs = dict(visual=lambda:visual(qmu,qb), acoustic=lambda:fp@qa,
                     fixed=lambda:.85*standardized(visual(qmu,qb))+.15*standardized(fp@qa),
                     b2=lambda:.85*standardized(visual(bmu,bb))+.15*standardized(fp@ba))
        saved = next(r for r in read(a.study/f'baselines/fields_seed42/{sid}.json') if r['query_id'] == qid)['methods']
        measurements = {}
        for name, fn in funcs.items():
            samples = []
            for rep in range(a.warmups+a.repeats):
                field, field_ms = timed(fn)
                ids, nms_ms = timed(lambda:proposals(field,xy))
                assert int(np.argmax(field)) == saved[name]['top1_id']
                np.testing.assert_array_equal(ids, saved[name]['candidate_ids'])
                if rep >= a.warmups:
                    samples.append(dict(field_ms=field_ms,score_nms_ms=nms_ms))
            measurements[name] = samples
        gp = georoot/'features_seed42'/hp.parent.name/(sid+'.npz')
        gr = read(gp.with_suffix('.json'))
        assert sha(gp) == gr['sha256'] and gr['candidate_sha256'] == hr['output_sha256']
        with np.load(gp) as z:
            assert int(z['query_ids'][i]) == qid
            S, V, ids = z['S'][i:i+1].copy(), z['V'][i:i+1].copy(), z['indices'][i].copy()
        reference = frozen_geo[sid,qid]
        np.testing.assert_array_equal(ids,reference['candidate_ids'])
        expected_gain = np.load(georoot/f'GEOQ_seed42/{hp.parent.name}_gains.npz')[sid][i:i+1]
        samples = []
        for rep in range(a.warmups+a.repeats):
            (choice,gain), ms = timed(lambda:geo.rank(model,S,V[...,:11],selection['selected']['threshold'],V[...,11:22]))
            np.testing.assert_allclose(gain,expected_gain,rtol=0,atol=1e-10)
            assert int(ids[choice[0]]) == reference['selected_id']
            if rep >= a.warmups:
                samples.append(dict(cached_candidate_rank_ms=ms))
        measurements['geoq'] = samples
        records.append(dict(scene=sid,query_id=qid,xy_candidates=len(xy),measurements=measurements,
                            top1_candidates_and_geoq_replay_exact=True,
                            adapted_sha256=sha(ap),candidate_sha256=sha(hp),geoq_features_sha256=sha(gp)))
        print('CHECKED',sid,qid,flush=True)
    result = dict(complete=True,smoke=a.smoke,queries=len(records),records=records,seed=42,grid_cm=20,
                  scope='Batch-one warm cached-input stages only; integration-only when smoke=true.',
                  exclusions=['RGB/RIR encoding and B2 feature adaptation','FP-RIR rendering/embedding','Support/refinement proposal construction and score-feature extraction','Loading, hashing, disk I/O, posterior diagnostics'],
                  warning='GEOQ timing starts from cached support candidates, not score-NMS outputs. Do not sum these timings into B2+GEOQ end-to-end latency. Shared hardware; no isolation established.',
                  selection='One uniform query per Val scene, RNG20261111, same manifest order as ray benchmark',
                  warmups=a.warmups,repeats=a.repeats,data_manifest_sha256=sha(a.study/'data_v1/manifest.json'),
                  baseline_math_sha256={n:sha(a.workspace/'AV-FPLoc/src/avfploc'/n) for n in ['geometry/rays.py','evaluation/baselines.py']},
                  source_sha256=sha(__file__),matching_sha256=sha(Path(__file__).with_name('matching.py')),
                  historical_geoq_sha256=sha(georoot/'inference.py'),model_sha256=sha(cp),audit_sha256=sha(auditp),
                  platform=platform.platform(),numpy_version=np.__version__,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__,joblib_version=joblib.__version__,threadpools=threadpoolctl.threadpool_info(),load_average=list(os.getloadavg()),
                  threads_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
                  test_accessed=False,training=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
