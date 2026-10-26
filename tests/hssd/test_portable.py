import json
from pathlib import Path
import numpy as np
import pytest
from avfploc.fusion.fixed import fuse, decode
from avfploc.fusion.features import extract, envelope
from avfploc.data.banks import load, aligned
from avfploc.evaluation.metrics import compare
from avfploc.vision.infer import score


@pytest.fixture
def banks():
    rng=np.random.default_rng(42)
    return rng.normal(size=(80,36)),rng.normal(size=(80,36)),np.c_[np.arange(80)*.2,np.zeros(80)]


def test_fixed_matches_original(banks):
    from avfploc.evaluation.baselines import standardized,select
    v,a,xy=banks
    np.testing.assert_array_equal(fuse(v,a,.1),.9*standardized(v)+.1*standardized(a))
    cell,yaw=select(v,a)[0]['equal_weight_fusion']
    np.testing.assert_array_equal(decode(fuse(v,a),xy),[*xy[cell],yaw*10])


@pytest.mark.parametrize('weight',[-1,1.1,np.nan])
def test_weight_rejected(banks,weight):
    with pytest.raises(ValueError):fuse(*banks[:2],weight)


@pytest.mark.parametrize('schema,n',[('core14',14),('processed58',58),('legacy17',17)])
def test_schema(banks,schema,n):
    v,a,xy=banks
    x,names=extract(v,a,xy,np.linspace(2,4,40),schema,np.ones((80,36))*3)
    assert len(x)==len(names)==n and np.isfinite(x).all() and len(set(names))==n


def test_envelope_winners(banks):
    v,a,_=banks
    for lo,hi,i in envelope(v,a):
        w=(lo+hi)/2
        assert i==np.argmax((1-w)*v+w*a)


def test_metrics():
    result=compare([[0,0,0],[2,0,90]],[[0,0,0],[0,0,0]],[[2,0,0],[0,0,0]])
    assert result['recall_1m']==.5 and result['repair_vs_fixed_xy']==1 and result['harm_vs_fixed_xy']==1


def test_bank_alignment(tmp_path,banks):
    v,a,xy=banks
    paths=[]
    for name,s in [('v',v),('a',a)]:
        p=tmp_path/(name+'.npz');np.savez(p,scores=s[None],xy=xy,query_ids=['q0'],yaw_deg=np.arange(36)*10);paths.append(p)
    vb,ab=map(load,paths);aligned(vb,ab)
    ab['query_ids']=np.array(['wrong'])
    with pytest.raises(ValueError):aligned(vb,ab)


def test_vision_geometry(banks):
    from avfploc.geometry.rays import angular_rays
    bank=np.arange(80*36).reshape(80,36)/100
    d=np.ones((1,40))*2;s=np.ones((1,40))
    rays,scales=angular_rays(d[0],s[0]);win=bank[:,(np.arange(36)[:,None]+np.arange(-5,6))%36]
    np.testing.assert_array_equal(score(d,s,bank)[0],-np.sum(abs(win-rays)/scales,axis=-1)/40)


def test_main_has_no_experiment_imports():
    import ast,avfploc
    for p in Path(avfploc.__file__).parent.rglob('*.py'):
        for n in ast.walk(ast.parse(p.read_text())):
            names=[n.module or ''] if isinstance(n,ast.ImportFrom) else [x.name for x in n.names] if isinstance(n,ast.Import) else []
            assert not any(x.startswith(('experiments','reliability_fusion','acoustic_branch')) for x in names),str(p)


def test_checkpoint_roundtrip(tmp_path,banks):
    torch=pytest.importorskip('torch')
    from avfploc.fusion.model import Gate
    from avfploc.fusion.infer import load_gate,predict
    v,a,xy=banks;x,names=extract(v,a,xy,np.ones(40),'core14')
    model=Gate(14,hidden=(128,64));p=tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(),mean=np.zeros(14),std=np.ones(14),feature_names=names,hidden=[128,64]),p)
    result,w=predict(v,a,xy,np.ones(40),load_gate(p),'core14')
    assert w==.5;np.testing.assert_array_equal(result,fuse(v,a))
    with pytest.raises(ValueError):predict(v,a,xy,np.ones(40),load_gate(p),'legacy17')


def test_training_rejects_overlap(tmp_path):
    pytest.importorskip('torch')
    from avfploc.fusion.train import run
    p=tmp_path/'manifest.json';p.write_text(json.dumps(dict(train=[dict(scene_id='same')],val=[dict(scene_id='same')])))
    with pytest.raises(ValueError,match='overlap'):run(p,tmp_path/'out')


def test_training_smoke(tmp_path,banks):
    pytest.importorskip('torch')
    from avfploc.fusion.train import run
    v,a,xy=banks;manifest={}
    for split in ('train','val'):
        folder=tmp_path/split;folder.mkdir()
        shared=dict(xy=xy,query_ids=np.array([split]),yaw_deg=np.arange(36)*10)
        np.savez(folder/'v.npz',scores=v[None],depth=np.ones((1,40))*2,**shared)
        np.savez(folder/'a.npz',scores=a[None],**shared)
        np.savez(folder/'t.npz',poses=np.array([[*xy[0],0]]),query_ids=shared['query_ids'])
        manifest[split]=[dict(scene_id=split,vision=f'{split}/v.npz',acoustic=f'{split}/a.npz',truth=f'{split}/t.npz')]
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    result=run(path,tmp_path/'run',epochs=1,schema='core14')
    assert result['epoch']==1 and (tmp_path/'run/best.pt').exists() and (tmp_path/'run/last.pt').exists()
    assert 'repair_vs_fixed_joint' in result['val']
