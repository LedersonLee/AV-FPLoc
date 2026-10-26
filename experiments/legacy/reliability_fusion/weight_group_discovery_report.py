"""Nested exploratory feature selector, clustered uncertainty, Train report.

Does not train/deploy a gate or select a weight for Test/Val. The small nested
selector controls one source of feature-screening optimism in the discovery.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy.special import softmax

from acoustic_branch.contracts import atomic_json,sha
from .input_supervision_audit import read,resolve,SOURCE,pose,yawdiff,REL as AUDIT
from .weight_group_discovery import REL,SEED,MASK_NAMES,get_tasks,metric,basin_features


def focused_analysis(out,root,ssd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from threadpoolctl import threadpool_limits
    path=out/'focused_analysis.json'
    if path.exists():return read(path)
    rows=read(out/'query_index.json');spec=read(out/'features.json');x=np.load(out/'features.npy');zs=[]
    refs={r['scene_id']:r for r in read(root/SOURCE/'scores_ready.json')['entries']if r['split']=='train'}
    for sid in dict.fromkeys(r['scene_id']for r in rows):
        xy=np.load(resolve(refs[sid],ssd)/'xy.npy');qs=read(root/'ReliabilityFusion/pilots/input_supervision_audit_v1/scenes'/sid/'queries.json')
        for q in qs:z,names=basin_features(q['parts'],xy);zs.append(z)
    z=np.array(zs);assert len(z)==len(x);xx=np.c_[x,z];fam=np.array(spec['families']);base=np.arange(14);old=np.arange(58);newcols=np.arange(x.shape[1],xx.shape[1])
    variants={'core14':base,'existing58':old,'basin_only':newcols,'core14+basin':np.r_[base,newcols],'existing58+basin':np.r_[old,newcols]}
    for family in ('prior_depth','prior_support','prior_response','prior_ray_fit'):
        cols=np.flatnonzero(fam==family);variants['core14+'+family]=np.r_[base,cols];variants['existing58-minus-'+family]=np.setdiff1d(old,cols)
    results=[];single=[]
    with threadpool_limits(limits=1):
      for task in ('exclusive3','all_fixed01_success'):
        ix,y,labels=get_tasks(rows)[task];groups=np.array([rows[i]['scene_id']for i in ix]);mat=xx[ix];probs={}
        for name,cols in variants.items():
            p=np.zeros((len(ix),len(labels)))
            for sid in np.unique(groups):
                a=groups!=sid;b=~a;sc=StandardScaler().fit(mat[a][:,cols]);m=LogisticRegression(C=.1,max_iter=1500,class_weight='balanced',random_state=SEED);m.fit(sc.transform(mat[a][:,cols]),y[a]);p[b]=m.predict_proba(sc.transform(mat[b][:,cols]))
            probs[name]=p;results.append(dict(task=task,variant=name,input_count=len(cols),**metric(y,p)));print('FOCUS',task,name,results[-1]['balanced_accuracy'],flush=True)
        ci=bootstrap_differences(y,[probs['existing58'],probs['existing58+basin']],groups)[1]
        next(r for r in results if r['task']==task and r['variant']=='existing58+basin')['delta_vs_existing58_ci95']=ci
        for j,name in enumerate(names):
            p=np.zeros((len(ix),len(labels)))
            for sid in np.unique(groups):
                a=groups!=sid;b=~a;m=DecisionTreeClassifier(max_depth=2,min_samples_leaf=25,class_weight='balanced',random_state=SEED);m.fit(z[ix][a,j:j+1],y[a]);p[b]=m.predict_proba(z[ix][b,j:j+1])
            single.append(dict(task=task,name=name,medians=[float(np.median(z[ix][y==k,j]))for k in range(len(labels))],**metric(y,p)))
    np.savez_compressed(out/'basin_features.npz',x=z,names=np.array(names))
    value=dict(results=results,single=single,definitions='Neighbourhood is around predicted fixed01 MAP, not GT. Large jump: >1m OR >30deg; contiguous stability ignores small grid drift. Exploratory follow-up on same Train scenes, no Val/Test.',feature_names=names)
    atomic_json(path,value);return value


def training_screen(x,y,k=5):
    """Class-balanced between/within contrast, FIT partition only."""
    means=np.array([x[y==c].mean(0)for c in np.unique(y)])
    within=np.array([x[y==c].var(0)for c in np.unique(y)]).mean(0)
    between=np.mean((means-means.mean(0))**2,axis=0)
    return np.argsort(-between/(within+1e-8),kind='stable')[:k]


def nested(task):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    out,taskname,kind=task;out=Path(out);dest=out/'nested'/f'{taskname}_{kind}.json';dest.parent.mkdir(exist_ok=True)
    if dest.exists():return read(dest)
    rows=read(out/'query_index.json');spec=read(out/'features.json');ix,y,labels=get_tasks(rows)[taskname]
    x=np.load(out/'features.npy',mmap_mode='r')[ix].copy();g=np.array([rows[i]['scene_id']for i in ix]);p=np.zeros((len(y),len(labels)));choices=[]
    with threadpool_limits(limits=1):
      for sid in np.unique(g):
        a=g!=sid;b=~a;cols=training_screen(x[a],y[a]);scale=StandardScaler().fit(x[a][:,cols]);xa=scale.transform(x[a][:,cols]);xb=scale.transform(x[b][:,cols])
        if kind=='linear':m=LogisticRegression(C=.1,max_iter=1500,class_weight='balanced',random_state=SEED);m.fit(xa,y[a])
        else:
            m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=25,l2_regularization=2.,early_stopping=False,random_state=SEED)
            count=np.bincount(y[a],minlength=len(labels));m.fit(xa,y[a],sample_weight=a.sum()/(len(labels)*count[y[a]]))
        p[b]=m.predict_proba(xb);choices.append(dict(held_out_scene=str(sid),features=[spec['names'][c]for c in cols]))
    result=dict(task=taskname,model=kind,variant='nested_top5',labels=labels,choices=choices,**metric(y,p))
    atomic_json(dest,result);np.savez_compressed(dest.with_suffix('.npz'),y=y,global_ids=ix,p=p);return result


def bootstrap_differences(y,probs,groups,repeats=2000):
    """Paired scene-cluster bootstrap of macro recall, fixed class set."""
    nclass=probs[0].shape[1];scenes=np.unique(groups);counts=[]
    for p in probs:
        pred=p.argmax(1);counts.append(np.array([np.bincount(nclass*y[groups==s]+pred[groups==s],minlength=nclass*nclass).reshape(nclass,nclass)for s in scenes]))
    counts=np.array(counts);rng=np.random.default_rng(SEED);draw=rng.integers(len(scenes),size=(repeats,len(scenes)))
    cs=counts[:,draw].sum(2);den=cs.sum(-1);good=np.all(den>0,axis=(0,2));cs=cs[:,good];den=den[:,good]
    ba=np.mean(np.diagonal(cs,axis1=-2,axis2=-1)/den,axis=-1);delta=ba-ba[0]
    return np.quantile(delta,[.025,.975],axis=1).T.tolist()


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL;report.mkdir(parents=True,exist_ok=True)
    result=read(out/'results.json');rows=read(out/'query_index.json');spec=read(out/'features.json');x=np.load(out/'features.npy')
    result['focused']=focused_analysis(out,root,ssd)
    tasks=[(str(out),t,k)for t in get_tasks(rows)for k in ('linear','tree')]
    with ProcessPoolExecutor(max_workers=6)as pool:
        nested_results=[f.result()for f in as_completed([pool.submit(nested,t)for t in tasks])]
    cis=[]
    for probe in result['probes']:
        task,kind=probe['task'],probe['model'];z=np.load(out/'probes'/f'{task}_{kind}.npz');ix=z['global_ids'];y=z['y'];g=np.array([rows[i]['scene_id']for i in ix])
        names=['core14','existing58','all'];preds=[z[n]for n in names]
        nz=np.load(out/'nested'/f'{task}_{kind}.npz');np.testing.assert_array_equal(nz['global_ids'],ix);names.append('nested_top5');preds.append(nz['p'])
        intervals=bootstrap_differences(y,preds,g)
        for n,ci in zip(names,intervals):cis.append(dict(task=task,model=kind,variant=n,delta_vs_core14_ci95=ci))
        intervals58=bootstrap_differences(y,[preds[1],preds[2],preds[3]],g)
        for n,ci in zip(['existing58','all','nested_top5'],intervals58):
            next(c for c in cis if c['task']==task and c['model']==kind and c['variant']==n)['delta_vs_existing58_ci95']=ci
    result['nested']=nested_results;result['paired_scene_bootstrap']=cis
    # Stable-set sensitivity: require actual success width within a bin.
    sensitivity=[]
    for min_width in (0.,.01,.025,.05):
        counts=np.zeros(8,int)
        for r in rows:
            widths=[sum(max(0,min(hi,b)-max(lo,a))for lo,hi in r['success_intervals'])for a,b in [(0,.1),(.1,.5),(.5,1)]]
            mask=sum(1<<i for i,v in enumerate(widths)if v>max(min_width,1e-12));counts[mask]+=1
        sensitivity.append(dict(minimum_bin_success_width=min_width,counts=counts.tolist()))
    result['success_width_sensitivity']=sensitivity
    # Two views: baseline-success controls and repaired-only ranges.
    controls=[]
    for name,ids in [('fixed01_success',[i for i,r in enumerate(rows)if r['baseline_success']]),('fixed01_fail_recoverable',[i for i,r in enumerate(rows)if not r['baseline_success']and r['mask']]),('unrecoverable',[i for i,r in enumerate(rows)if not r['mask']])]:
        controls.append(dict(name=name,queries=len(ids),feature_medians=dict(zip(spec['names'],np.median(x[ids],axis=0).tolist())),median_success_width=float(np.median([rows[i]['success_width']for i in ids])),median_components=float(np.median([rows[i]['success_components']for i in ids]))))
    result['baseline_controls']=controls
    distribution_stats={name:[np.quantile(x[np.array([r['mask']==m for r in rows]),j],[.1,.25,.5,.75,.9]).tolist()for m in range(8)]for j,name in enumerate(spec['names'])}
    atomic_json(report/'distribution_stats.json',distribution_stats)
    # Compact representative gallery: one medoid per feasible mask, plus baseline controls.
    scale=np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0),1e-5)
    selected=[]
    for mask in range(8):
        for fixed in (True,False):
            ids=np.array([i for i,r in enumerate(rows)if r['mask']==mask and r['baseline_success']==fixed])
            if not len(ids):continue
            xx=x[ids]/scale;centre=np.median(xx,axis=0);order=ids[np.argsort(np.mean((xx-centre)**2,axis=1))]
            selected.extend(order[:2].tolist())
    refs={r['scene_id']:r for r in read(root/SOURCE/'scores_ready.json')['entries']if r['split']=='train'}
    gallery=[]
    for d in ('rgb','maps','heatmaps'):(report/d).mkdir(exist_ok=True)
    for i in selected:
        r=rows[i];sid=r['scene_id'];source=resolve(refs[sid],ssd);queries=read(source/'queries.json');lut={q['sample_index']:j for j,q in enumerate(queries)}
        v,a=np.load(source/'scores.npy',mmap_mode='r')[lut[r['sample_index']]];w=r['oracle_weight']
        layers=np.array([s.max(1)for s in(v,a,.9*v+.1*a,(1-w)*v+w*a)],dtype='<f4');name=sid+'_'+str(r['sample_index'])
        layers.tofile(report/'heatmaps'/(name+'.bin'));shutil.copy2(ssd/r['image_path'],report/'rgb'/(name+'.png'))
        shutil.copy2(root/'2DFP'/sid/'map.png',report/'maps'/(sid+'.png'))
        aq=read(root/AUDIT/'scenes'/sid/'queries.json');source_q=next(q for q in aq if q['sample_index']==r['sample_index'])
        xy=np.load(source/'xy.npy');base=np.array(r['methods']['fixed01']['pose']);trace=[]
        for lo,hi,idx in source_q['parts']:
            p=pose(int(idx),xy);trace.append(dict(lo=lo,hi=hi,xy_from_fixed=float(np.linalg.norm(p[:2]-base[:2])),yaw_from_fixed=float(yawdiff(p[2],base[2]))))
        gallery.append(dict(index=i,**r,image='rgb/'+name+'.png',heatmap='heatmaps/'+name+'.bin',weight_trace=trace))
    banks={sid:dict(xy=np.load(resolve(refs[sid],ssd)/'xy.npy').tolist(),metadata=read(root/'2DFP'/sid/'floorplan_metadata.json'))for sid in {r['scene_id']for r in gallery}}
    atomic_json(report/'gallery.json',gallery);atomic_json(report/'banks.json',banks)
    # Feature arrays remain downloadable, all6387 included; GT diagnostics separated.
    np.savez_compressed(out/'discovery_features.npz',x=x,names=np.array(spec['names']),scene=np.array([r['scene_id']for r in rows]),sample=np.array([r['sample_index']for r in rows]),mask=np.array([r['mask']for r in rows]),fixed01_success=np.array([r['baseline_success']for r in rows]))
    with (out/'all_queries.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['scene_id','sample_index','pose_id','feasible_mask','fixed01_success','success_width','success_components',*spec['names']])
        for r,xx in zip(rows,x):writer.writerow([r['scene_id'],r['sample_index'],r['pose_id'],r['mask_name'],r['baseline_success'],r['success_width'],r['success_components'],*xx.tolist()])
    atomic_json(out/'report_results.json',result);atomic_json(report/'results.json',result)
    for name in ('all_queries.csv','features.json','discovery_features.npz','basin_features.npz'):shutil.copy2(out/name,report/name)
    shutil.copy2(Path(__file__).with_name('weight_group_discovery_report.html'),report/'index.html')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.4),constrained_layout=True)
    g=result['groups'];labels=[r['name']for r in g];b=np.array([r['fixed01_success']for r in g]);n=np.array([r['queries']for r in g]);pos=np.arange(8)
    axes[0].barh(pos,b,label='Fixed .1 success',color='#3788ba');axes[0].barh(pos,n-b,left=b,label='Fixed .1 fail',color='#da9845');axes[0].set(yticks=pos,yticklabels=labels,xlabel='Train queries (all6387)',title='Full feasible-bin masks, no arbitrary weight assignment');axes[0].legend()
    p=next(z for z in result['probes']if z['task']=='exclusive3'and z['model']=='linear');d={r['variant']:r for r in p['results']}
    keys=['core14','existing58','new_only','all','only_competition','only_geometry','only_rgb'];axes[1].barh(np.arange(len(keys)),[d[k]['balanced_accuracy']*100 for k in keys],color='#537fac');axes[1].axvline(100/3,color='gray',linestyle='--');axes[1].set(yticks=np.arange(len(keys)),yticklabels=keys,xlim=(0,100),xlabel='Scene-held-out balanced accuracy (%)',title='Exclusive low / medium / high (n=484)')
    fig.savefig(report/'summary.png',dpi=150);plt.close(fig)
    verification=dict(status='passed',train_queries=6387,scenes=12,features=len(spec['names']),feature_array_sha256=sha(out/'features.npy'),results_sha256=sha(out/'report_results.json'),test_accessed=False,val_accessed=False,source_code_sha256=sha(Path(__file__).with_name('weight_group_discovery.py')),diagnostic_models_only=True)
    atomic_json(out/'verification.json',verification);atomic_json(report/'verification.json',verification);print('REPORT COMPLETE',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())
