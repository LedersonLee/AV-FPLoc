import math
import pytest
import torch
from track_b.scripts.summarize_ray_uncertainty import paired_scene,ray_stats


def test_scene_bootstrap_does_not_count_query_replication_as_new_scenes():
    def data(n):
        a=[];b=[]
        for sid in ['a','b']:
            for seed in [42,43,44]:
                for q in range(n):
                    meta=dict(scene=sid,seed=seed,query=q)
                    a.append(dict(**meta,joint=float(sid=='b')));b.append(dict(**meta,joint=0.))
        return a,b
    a,b=data(2);x=paired_scene(a,b,['a','b'],'joint')
    aa,bb=data(20);y=paired_scene(aa,bb,['a','b'],'joint')
    assert x==y and x['delta']==50 and x['ci95']==[0,100]
    with pytest.raises(AssertionError):paired_scene(a+a[:1],b,['a','b'],'joint')
    with pytest.raises(AssertionError):paired_scene(a,b[:-1],['a','b'],'joint')


def test_ray_stats_analytical_constant_scale():
    records=[dict(mean=torch.zeros(16),target=torch.ones(16)*.1,scale=torch.ones(16)*.1,visible=torch.ones(16)) for _ in range(10)]
    result={x['stratum']:x for x in ray_stats(records)};v=result['visible']
    assert v['count']==160 and abs(v['nll']-(1+math.log(.2)))<1e-6
    assert v['coverage']['0.5']==0 and v['coverage']['0.8']==1
    assert v['pearson_scale_error'] is None and result['outside_fov']['count']==0
