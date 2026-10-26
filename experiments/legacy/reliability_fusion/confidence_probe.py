"""A diagnostic extension: preserve predicted visual uncertainty as input.

Train-only fixed probes, val-only reporting. Does not alter the trained gate.
"""
import argparse
import html
from pathlib import Path

import numpy as np

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from .prepare import REL
from .design_probe import probe,metrics,subset
from .audit import auc


def run(root,out,report):
    root,out,report=map(Path,(root,out,report));base=root/REL
    ready=read(base/'scores_ready.json');scene_rows=[];sources={}
    if ready['test_accessed']:raise ValueError('Test forbidden')
    for entry in ready['entries']:
        if entry['split'] not in ('train','val'):raise ValueError('Test forbidden')
        scene=entry['scene_id'];cache=out/'scenes'/(scene+'.npz')
        with np.load(cache) as blob:row={k:blob[k].copy() for k in blob.files}
        path=base/'vision'/entry['split']/(scene+'.npz');receipt=read(path.with_suffix('.json'))
        if sha(path)!=receipt['data_sha256']:raise ValueError('Frozen visual uncertainty changed')
        queries=read(root/entry['directory']/'queries.json')
        with np.load(path) as blob:
            if list(blob['sample_indices'])!=[q['sample_index'] for q in queries]:raise ValueError('Query order mismatch')
            location,scale=blob['location'],blob['scale']
            if not (scale>0).all() or not np.isfinite(scale).all():raise ValueError('Invalid predicted scale')
            extra=np.column_stack([scale.mean(1),scale.mean(1)/np.maximum(abs(location).mean(1),.1),
                                   scale.std(1)/np.maximum(scale.mean(1),1e-6)])
        row['features']=np.column_stack([row['features'],extra]);scene_rows.append(row)
        sources[scene]=dict(vision_sha256=sha(path),curve_sha256=sha(cache))
    data={k:np.concatenate([r[k] for r in scene_rows]) for k in scene_rows[0]}
    train=subset(data,data['split']=='train');val=subset(data,data['split']=='val')
    results={}
    for name,cols in [('original17',list(range(17))),('original17_plus_scale3',list(range(20))),('scale3_only',[17,18,19])]:
        for kind in ('ridge','knn'):
            chosen=probe(train,val,cols,kind);results[name+'_'+kind]=metrics(val,chosen)
    output=dict(status='complete',test_accessed=False,production_retraining=False,
        added_features=['visual_scale_mean','visual_scale_over_depth_mean','visual_scale_cv'],
        train_queries=len(train['scene']),val_queries=len(val['scene']),sources=sources,results=results,
        warning='Fixed diagnostic probes; no validation tuning. This is not the deployed continuous gate.')
    output['endpoint_and_fusion_oracles']={}
    output['added_feature_auc']={}
    for split,rows in [('train',train),('val',val)]:
        v,a=rows['ok'][:,0],rows['ok'][:,-1]
        success=rows['ok'];padded=np.pad(success.astype(int),((0,0),(1,1)))
        runs=(np.diff(padded,axis=1)==1).sum(1)
        viable=success.any(1)
        average=np.rint((success*np.arange(21)).sum(1)/np.maximum(success.sum(1),1)).astype(int)
        output['endpoint_and_fusion_oracles'][split]=dict(queries=len(v),
            endpoint_oracle_recall_1m=float((v|a).mean()),
            any_scalar_weight_oracle_recall_1m=float(rows['ok'].any(1).mean()),
            both_endpoints_fail_but_some_weight_succeeds=int((~v&~a&rows['ok'].any(1)).sum()),
            disjoint_success_weight_intervals=int((runs>1).sum()),
            mean_of_successful_weights_is_unsuccessful=int((viable&~success[np.arange(len(v)),average]).sum()))
        exclusive=v!=a
        output['added_feature_auc'][split]=dict(exclusive_queries=int(exclusive.sum()),
            positive='acoustic-only correct, vision-only wrong',
            values=[auc(rows['features'][exclusive,i],a[exclusive]) for i in (17,18,19)])
    atomic_json(out/'confidence_extension.json',output)
    atomic_json(report/'confidence_extension.json',output)
    rows=[]
    for name,m in results.items():
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (name,f'{100*m["recall_1m"]:.2f}',f'{100*m["recall_1m_30deg"]:.2f}',f'{m["mean_weight"]:.3f}'))+'</tr>')
    page=report/'index.html'
    prefix=page.read_text().split('<h2>Predicted uncertainty input extension')[0]
    page.write_text(prefix+'''<h2>Predicted uncertainty input extension (diagnostic only)</h2>
<p>Add mean Laplace scale, scale/depth ratio, scale CV from frozen visual predictions.
Same train-only fixed ridge/kNN probes and untouched held-out val labels. No production gate retraining.</p>
<table><tr><th>Probe</th><th>Val R@1m %</th><th>Val R@1m,30° %</th><th>Mean acoustic weight</th></tr>'''+''.join(rows)+
        '</table><p><a href="confidence_extension.json">Extension inputs and results</a></p>')
    print(output['results'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out','report'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.out,a.report)
