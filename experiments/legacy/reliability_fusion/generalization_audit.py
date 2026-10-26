"""Frozen 128/64, processed58 Train/Val support and label-consistency audit.

No model fitting or Test references. Train nearest neighbours always exclude
the query's entire scene. GT success sets are diagnostic labels, not distances.
"""
import argparse
import csv
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance

from acoustic_branch.contracts import atomic_json, sha
from .capacity_probe import REL as CAPACITY
from .checkpoint_drift import REL as DRIFT, merged, component, transition, logit
from .input_supervision_audit import read, intersection, measure
from .response_input_localization import REL as DATA, load_scene

REL = 'ReliabilityFusion/analysis/generalization_58_v1'
FAMILIES = {'core14':list(range(14)), 'depth4':list(range(14,18)),
            'support14':list(range(18,32)), 'response14':list(range(32,46)),
            'ray_fit12':list(range(46,58))}


def empirical_rank(x, train):
    """Train mid-CDF for ties; bounded and insensitive to raw feature tails."""
    result=np.empty_like(x,dtype=float)
    for j in range(x.shape[1]):
        a=np.sort(x[train,j])
        result[:,j]=(np.searchsorted(a,x[:,j],side='left')+np.searchsorted(a,x[:,j],side='right'))/(2*len(a))
    return result


def neighbour_indices(x, train, groups, k=20):
    ti=np.flatnonzero(train)
    d=cdist(x,x[train])/np.sqrt(x.shape[1])
    d[groups[:,None]==groups[ti][None,:]]=np.inf
    # Stable ties are owned by Train row ID, including at the k-th boundary.
    ix=np.argsort(d,axis=1,kind='stable')[:,:k]
    return ti[ix],np.take_along_axis(d,ix,axis=1)


def interval_pair(a,b):
    both=bool(a and b);inter=intersection(a,b) if both else 0.
    return dict(both_feasible=both,disjoint=bool(both and inter<=1e-12),
        jaccard=inter/(measure(a)+measure(b)-inter) if both else None)


def profile(values):
    a=np.asarray(values,float)
    return dict(n=len(a),min=float(a.min()),max=float(a.max()),mean=float(a.mean()),p10=float(np.quantile(a,.1)),p25=float(np.quantile(a,.25)),median=float(np.median(a)),p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95))) if len(a) else dict(n=0)


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    cfg=read(root/CAPACITY/'config.json');ready=read(root/DATA/'ready.json')
    assert sha(root/DATA/'ready.json')==cfg['data_ready_sha256']
    refs=sorted(ready['scenes'],key=lambda r:r['scene_id'])
    assert {r['split'] for r in refs}=={'train','val'}
    scenes={};rows=[];xs=[];scenes_hash={}
    for ref in refs:
        sid=ref['scene_id'];assert sid in cfg['selection'][ref['split']]
        folder=root/DATA/'scenes'/sid
        assert sha(folder/'queries.json')==ref['queries_sha256']
        s=load_scene(root/DATA,sid,root,ssd);scenes[sid]=s;xs.append(s['x'][:,:58])
        scenes_hash[sid]=dict(data_sha256=ref['data_sha256'],queries_sha256=ref['queries_sha256'])
        for i,q in enumerate(s['queries']):
            rows.append(dict(scene_id=sid,split=s['split'],sample_index=q['sample_index'],pose_id=q['pose_id'],
                row=i,image_path=q['image_path'],success=merged(q['success']['joint']),valid=bool(s['valid'][i]),
                target=float(s['interior_target'][i]) if s['valid'][i] else None))
    x=np.concatenate(xs);train=np.array([r['split']=='train' for r in rows]);groups=np.array([r['scene_id'] for r in rows]);n=len(rows)
    assert train.sum()==6387 and (~train).sum()==1704 and x.shape==(8091,58)
    assert len(set(groups[train]))==12 and len(set(groups[~train]))==3
    names=refs[0]['features'][:58];mean=x[train].mean(0);std=x[train].std(0);std=np.where(std>1e-6,std,1.)
    z=(x-mean)/std;rank=empirical_rank(x,train)
    # Equal contribution per interpretable family, using exactly the same 58
    # inputs. This is a diagnostic metric control, not a new MLP normalization.
    scale=np.ones(58)
    for cols in FAMILIES.values():scale[cols]=np.sqrt(58/(len(FAMILIES)*len(cols)))
    representations={'standard58':z,'rank_family58':rank*scale}
    neighbours={};distances={};support={};pairs={};query_details={}
    for metric,rep in representations.items():
        ids,ds=neighbour_indices(rep,train,groups);neighbours[metric]=ids;distances[metric]=ds
        assert not np.any(groups[ids]==groups[:,None])
        assert np.isfinite(ds).all()
        for i in [0,100,2000,5000,n-1]:
            pool=np.flatnonzero(train&(groups!=groups[i]))
            direct=np.sqrt(np.mean((rep[pool]-rep[i])**2,axis=1))
            assert ids[i,0]==pool[np.argmin(direct)]
            np.testing.assert_allclose(ds[i,0],direct.min(),rtol=1e-12,atol=1e-12)
        thresholds={f'p{p}':float(np.quantile(ds[train,0],p/100)) for p in(25,50,75,95,99)}
        details=[]
        for i,ns in enumerate(ids):
            ps=[interval_pair(rows[i]['success'],rows[j]['success']) for j in ns]
            first=ps[0];eligible=[p for p in ps[:5] if p['both_feasible']]
            j=int(ns[0]);both_valid=rows[i]['valid'] and rows[j]['valid']
            details.append(dict(neighbour=int(ns[0]),distance=float(ds[i,0]),both_feasible=first['both_feasible'],
                conflict=first['disjoint'],jaccard=first['jaccard'],
                k5_conflict=float(np.mean([p['disjoint'] for p in eligible])) if eligible else None,
                both_supervised=both_valid,
                target_logit_delta=abs(float(logit(rows[i]['target'])-logit(rows[j]['target']))) if both_valid else None,
                neighbour_target_safe=(component(rows[i]['success'],rows[j]['target']) is not None) if rows[j]['valid'] else None,
                query_target_safe_for_neighbour=(component(rows[j]['success'],rows[i]['target']) is not None) if rows[i]['valid'] else None,
                ood95=bool(ds[i,0]>thresholds['p95']),close25=bool(ds[i,0]<=thresholds['p25'])))
        query_details[metric]=details;support[metric]={'thresholds_from_train_cross_scene':thresholds}
        for split,mask in [('train_cross_scene',train),('val_to_train',~train)]:
            pp=[];nn=[];bins=[]
            for i in np.flatnonzero(mask):
                for j in ids[i]:pp.append(interval_pair(rows[i]['success'],rows[j]['success']))
                nn.append(details[i])
            def pair_summary(rr):
                eligible=[p for p in rr if p['both_feasible']]
                return dict(pairs=len(rr),both_feasible=len(eligible),disjoint=sum(p.get('disjoint',p.get('conflict',False)) for p in eligible),
                    disjoint_rate=float(np.mean([p.get('disjoint',p.get('conflict',False)) for p in eligible])) if eligible else None,
                    jaccard_mean=float(np.mean([p['jaccard'] for p in eligible])) if eligible else None)
            support[metric][split]=dict(distance=profile(ds[mask,0]),ood95=int(sum(p['ood95'] for p in nn)),
                close25=int(sum(p['close25'] for p in nn)),nearest=pair_summary(nn),k20=pair_summary(pp))
            supervised=[p for p in nn if p['both_supervised']]
            compat=[p for p in supervised if p['both_feasible'] and not p['conflict']]
            support[metric][split]['point_targets']=dict(both_supervised=len(supervised),
                actual_success_conflicts=sum(p['conflict'] for p in supervised),
                compatible=len(compat),
                compatible_but_at_least_one_target_unsafe=sum(not p['neighbour_target_safe'] or not p['query_target_safe_for_neighbour'] for p in compat),
                logit_delta=profile([p['target_logit_delta'] for p in supervised]))
            support[metric][split]['per_scene']={sid:dict(queries=int(np.sum(groups==sid)),
                distance=profile(ds[groups==sid,0]),ood95=int(sum(details[i]['ood95'] for i in np.flatnonzero(groups==sid))))
                for sid in sorted(set(groups[mask]))}
            for label,condition in [('close_train_q25',lambda p:p['close25']),('in_support95',lambda p:not p['ood95']),('ood95',lambda p:p['ood95'])]:
                bins.append(dict(band=label,**pair_summary([p for p in nn if condition(p)])))
            support[metric][split]['bands']=bins
    # Feature shifts use Train-only scaling; scene-held-out Train comparisons
    # provide context, not IID null p-values (nearby poses are correlated).
    train_groups=sorted(set(groups[train]));feature_rows=[]
    for j,name in enumerate(names):
        a,b=x[train,j],x[~train,j];lo,hi=np.quantile(a,[.01,.99]);scene_ks=[]
        for sid in train_groups:
            scene_ks.append(float(ks_2samp(x[groups==sid,j],x[train&(groups!=sid),j]).statistic))
        ks=float(ks_2samp(a,b).statistic)
        val_scenes={sid:dict(mean_z=float(z[groups==sid,j].mean()),ks=float(ks_2samp(a,x[groups==sid,j]).statistic),
            outside_train_p1_p99=float(np.mean((x[groups==sid,j]<lo)|(x[groups==sid,j]>hi)))) for sid in sorted(set(groups[~train]))}
        feature_rows.append(dict(index=j,name=name,family=next(f for f,c in FAMILIES.items() if j in c),
            train_mean=float(a.mean()),train_std=float(std[j]),val_mean=float(b.mean()),mean_shift_z=float(z[~train,j].mean()),
            ks=ks,wasserstein_over_train_std=float(wasserstein_distance(a,b)/std[j]),
            outside_train_p1_p99=float(np.mean((b<lo)|(b>hi))),
            train_scene_ks_median=float(np.median(scene_ks)),train_scene_ks_max=max(scene_ks),
            val_ks_above_all_train_scenes=ks>max(scene_ks),val_scenes=val_scenes))
    # Bind to prior fully verified two-checkpoint predictions. Train/Val only.
    drift_report=ssd/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/fusion-checkpoint-drift'
    dv=read(root/DRIFT/'verification.json');assert dv['status']=='passed' and not dv['test_accessed']
    assert sha(drift_report/'records.json')==dv['records_sha256']
    records=read(drift_report/'records.json');lookup={(r['seed'],r['scene_id'],r['sample_index']):r for r in records}
    analyses=[];exports=[]
    for metric,ds in query_details.items():
        for seed in(42,43):
            dr=[lookup[(seed,r['scene_id'],r['sample_index'])] for r in rows]
            target=np.array([r['target'] if r['valid'] else np.nan for r in rows])
            wb=np.array([r['weight_best'] for r in dr]);wl=np.array([r['weight_last'] for r in dr])
            for split,mask in [('train_cross_scene',train),('val_to_train',~train)]:
                bands=[]
                specs=[('all',lambda d:True),('close_train_q25',lambda d:d['close25']),
                    ('in_support95',lambda d:not d['ood95']),('ood95',lambda d:d['ood95']),
                    ('feasible_conflict',lambda d:d['both_feasible'] and d['conflict']),
                    ('feasible_compatible',lambda d:d['both_feasible'] and not d['conflict']),
                    ('close_conflict',lambda d:d['close25'] and d['both_feasible'] and d['conflict']),
                    ('close_compatible',lambda d:d['close25'] and d['both_feasible'] and not d['conflict'])]
                for label,condition in specs:
                    ix=[i for i in np.flatnonzero(mask) if condition(ds[i])]
                    if not ix:continue
                    bb=np.array([dr[i]['states']['best']['joint'] for i in ix]);ll=np.array([dr[i]['states']['last']['joint'] for i in ix]);vv=[i for i in ix if rows[i]['valid']]
                    harm=int(np.sum(bb&~ll));repair=int(np.sum(~bb&ll));eligible=int(bb.sum())
                    losses={phase:float(np.mean([dr[i]['loss_'+phase] for i in vv])) if vv else None for phase in('best','last')}
                    # Measure how much last fits the held-out query vs its
                    # closest Train neighbour; the neighbour itself never uses
                    # Val labels and can be unsupervised.
                    nn=[int(ds[i]['neighbour']) for i in ix];npv=[j for j in nn if rows[j]['valid']]
                    bands.append(dict(band=label,queries=len(ix),best_joint=float(bb.mean()),last_joint=float(ll.mean()),
                        harm=harm,repair=repair,best_success=eligible,harm_rate_given_best_success=harm/eligible if eligible else None,
                        loss_best=losses['best'],loss_last=losses['last'],distance_median=float(np.median([ds[i]['distance'] for i in ix])),
                        nn_train_loss_best=float(np.mean([dr[j]['loss_best'] for j in npv])) if npv else None,
                        nn_train_loss_last=float(np.mean([dr[j]['loss_last'] for j in npv])) if npv else None,
                        median_logit_query_nn_best=float(np.median(abs(logit(wb[ix])-logit(wb[nn])))),
                        median_logit_query_nn_last=float(np.median(abs(logit(wl[ix])-logit(wl[nn]))))))
                harms=[i for i in np.flatnonzero(mask) if dr[i]['joint_transition']=='harm']
                cause_counts=dict(total=len(harms),ood95=sum(ds[i]['ood95'] for i in harms),
                    close25=sum(ds[i]['close25'] for i in harms),
                    both_feasible_conflict=sum(ds[i]['both_feasible'] and ds[i]['conflict'] for i in harms),
                    both_feasible_compatible=sum(ds[i]['both_feasible'] and not ds[i]['conflict'] for i in harms),
                    no_pair_feasibility=sum(not ds[i]['both_feasible'] for i in harms),
                    close_conflict=sum(ds[i]['close25'] and ds[i]['both_feasible'] and ds[i]['conflict'] for i in harms),
                    both_supervised=sum(ds[i]['both_supervised'] for i in harms),
                    neighbour_target_would_succeed=sum(ds[i]['both_supervised'] and ds[i]['neighbour_target_safe'] for i in harms))
                analyses.append(dict(metric=metric,seed=seed,split=split,bands=bands,harm_counts=cause_counts))
            if metric=='standard58':
                for i,r in enumerate(rows):
                    d=ds[i];j=d['neighbour'];nr=rows[j]
                    exports.append(dict(seed=seed,**r,**d,neighbour_scene=nr['scene_id'],neighbour_sample=nr['sample_index'],
                        neighbour_success=nr['success'],neighbour_target=nr['target'],
                        transition=dr[i]['joint_transition'],weight_best=wb[i],weight_last=wl[i],
                        nn_weight_best=wb[j],nn_weight_last=wl[j],loss_best=dr[i]['loss_best'],loss_last=dr[i]['loss_last']))
    # Most similar contradictory and compatible pairs. Include both Train
    # cross-scene and Val->Train; select by distance, not by desired conclusion.
    gallery=[];seen=set()
    for split in('val','train'):
        for typ in('close_conflict','compatible_harm','ood_harm'):
            candidates=[r for r in exports if r['seed']==42 and r['split']==split and
                ((r['both_feasible'] and r['conflict']) if typ=='close_conflict' else
                 (r['transition']=='harm' and r['both_feasible'] and not r['conflict']) if typ=='compatible_harm' else
                 (r['transition']=='harm' and r['ood95']))]
            for r in sorted(candidates,key=lambda r:r['distance']):
                key=(r['scene_id'],r['sample_index'])
                if key in seen:continue
                seen.add(key);i=next(i for i,q in enumerate(rows) if(q['scene_id'],q['sample_index'])==key);j=r['neighbour']
                gallery.append(dict(type=typ,query_index=i,neighbour_index=j,query=r,neighbour=rows[j],
                    features=[dict(name=names[k],query=float(x[i,k]),neighbour=float(x[j,k]),delta_z=float(z[i,k]-z[j,k])) for k in np.argsort(-abs(z[i]-z[j]))[:10]]))
                if sum(g['type']==typ and g['query']['split']==split for g in gallery)>=2:break
    (report/'rgb').mkdir(exist_ok=True)
    for g in gallery:
        for key in('query','neighbour'):
            r=g[key];name=r['scene_id']+'_'+str(r['sample_index'])+'.png';shutil.copy2(ssd/r['image_path'],report/'rgb'/name);r['image']='rgb/'+name
        # Last/Best outputs of the actual same frozen model at the neighbour.
        nr=lookup[(42,g['neighbour']['scene_id'],g['neighbour']['sample_index'])]
        g['neighbour'].update(weight_best=nr['weight_best'],weight_last=nr['weight_last'])
    feature_rows.sort(key=lambda r:-r['ks'])
    result=dict(scope=dict(train_scenes=12,val_scenes=3,train_queries=6387,val_queries=1704),
        model=dict(hidden=[128,64],features=58,seeds=[42,43],updated=False),
        support=support,features=feature_rows,families=FAMILIES,analyses=analyses,
        metric_notes=dict(standard58='RMS Euclidean of exactly the Train-standardized 58 inputs used by MLP.',
            rank_family58='Same 58 features, Train empirical ranks with equal family contribution. Diagnostic sensitivity control; MLP normalization unchanged.'),
        interpretation=dict(
            distribution='No strong global covariate-support shift detected by either fixed metric: Val distances/OOD fractions do not exceed Train-cross-scene reference. Marginal KS shifts are <=.131, and none exceeds the largest held-out Train-scene KS. This does NOT rule out conditional shift or shifts invisible to these metrics.',
            input_limitation='Real incompatible-success pairs exist, but nearest-pair conflict rates are not higher in Val than cross-scene Train. Only 6/58 and 6/52 Val Harm use conflicting nearest feasible Train labels in primary metric. Similar is not identical; this is not proof of irreducible ambiguity.',
            overfitting='Most Val Harm occurs in nominal support, often with compatible Train-neighbour success sets. Actual Train-neighbour MSE improves while Val MSE worsens; median query-neighbour output difference grows. This favours overfitting/unstable learned mapping as the immediate explanation, not severe global OOD or a universal lack of usable information.',
            example='Val 108736737_177263406/Q1234 and Train 102816729/Q1558 have overlapping success [0,.4135] and targets .2068/.2348; Best weights .3632/.3447 both work, Last .5299/.1902 fails only Val. Shared valid weights exist, so this failure does not force contradictory outputs.',
            next_control='Keep 128/64, 58 inputs and logit targets fixed. Diagnose early stopping and regularization with scene-grouped Train validation before changing inputs or supervision. No new training or Test evaluation was performed.'),
        caveats=['Cross-scene nearest neighbours only; no same-scene adjacent yaw shortcuts.',
            'Disjoint means both queries have nonempty continuous Joint success sets with no positive-width intersection. It does NOT prove identical inputs or irreducible ambiguity.',
            'Different point targets with overlapping success sets are not an incompatible localization requirement.',
            'Train-cross-scene 95th percentile defines OOD; this is one distance-based support diagnostic, not a universal OOD certificate.',
            'Train/Val scene-specific KS is descriptive, not IID significance testing; neighbouring poses are correlated.',
            'Val already selected Best and has been repeatedly inspected. Only three Val scenes; no causal proof or independent Test claim.'],
        test_accessed=False)
    atomic_json(out/'summary.json',result);atomic_json(report/'summary.json',result)
    atomic_json(report/'gallery.json',gallery);atomic_json(report/'queries.json',exports)
    atomic_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),features=names,fit='Train only'))
    np.savez_compressed(out/'neighbours.npz',**{m+'_indices':v for m,v in neighbours.items()},**{m+'_distances':v for m,v in distances.items()})
    with(report/'features.csv').open('w')as f:
        keys=[k for k in feature_rows[0] if k!='val_scenes'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(feature_rows)
    with(report/'queries.csv').open('w')as f:
        keys=['seed','split','scene_id','sample_index','neighbour_scene','neighbour_sample','distance','ood95','close25','both_feasible','conflict','jaccard','k5_conflict','both_supervised','target_logit_delta','neighbour_target_safe','query_target_safe_for_neighbour','transition','weight_best','weight_last','target','neighbour_target','nn_weight_best','nn_weight_last','loss_best','loss_last']
        w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(exports)
    checked=dict(status='passed',test_accessed=False,scope=result['scope'],model_updated=False,inputs_changed=False,
        train_only_normalization=True,all_train_neighbours_exclude_entire_scene=True,
        data_ready_sha256=sha(root/DATA/'ready.json'),source_scenes=scenes_hash,
        prior_verified_predictions_sha256=sha(drift_report/'records.json'),source_sha256=sha(Path(__file__)))
    atomic_json(out/'verification.json',checked);atomic_json(report/'verification.json',checked)
    shutil.copy2(Path(__file__).with_suffix('.html'),report/'index.html')
    print('SUPPORT',support,flush=True)
    for a in analyses:print('HARM',a['metric'],a['seed'],a['split'],a['harm_counts'],flush=True)
    print('REPORT',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())
