"""TRAIN-only observational analysis of oracle-repaired weight groups.

No fitting, new RIRs, Val/Test reads, or gate changes. Grouping a chosen oracle
weight is not proof that that weight range is necessary: full stable success
intervals are reported too. Canonical scores/GT are never replaced or edited.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from acoustic_branch.contracts import atomic_json, immutable_json, sha
from .input_supervision_audit import SOURCE, REL as AUDIT, read, resolve, pose, yawdiff, intersection, measure

REL='ReliabilityFusion/analysis/oracle_weight_groups_v1'
ORACLE='ReliabilityFusion/pilots/independent_pose_risk_v1'
BINS={'low':(0.,.1),'medium':(.1,.5),'high':(.5,1.)}
METHODS=['vision','acoustic','fixed01','oracle']


def group_weight(w):
    return 'low' if w<.1 else 'medium' if w<.5 else 'high'


def success_support(intervals):
    return {g:intersection(intervals,[[lo,hi]])>1e-12 for g,(lo,hi) in BINS.items()}


def zscore(s):
    return (s-s.mean())/s.std() if s.std()>1e-12 else np.zeros_like(s)


def modality_stats(score,xy,truth):
    idx=int(score.argmax()); p=pose(idx,xy); d=float(np.linalg.norm(p[:2]-truth[:2])); angle=float(yawdiff(p[2],truth[2]))
    spatial=score.max(1); z=zscore(spatial); prob=softmax(z); top=np.argsort(-spatial,kind='stable')[:50]
    around=np.linalg.norm(xy-xy[idx//36],axis=1)<=1
    gtxy=np.linalg.norm(xy-truth[:2],axis=1)<=1; gtyaw=yawdiff(np.arange(36)*10,truth[2])<=30
    good=gtxy[:,None]&gtyaw[None,:]; bestgt=float(score[good].max())
    full=softmax(score.ravel()).reshape(score.shape); yawprob=softmax(zscore(score[idx//36]))
    entropy=lambda p:float(-(p*np.log(np.maximum(p,1e-300))).sum()/np.log(len(p)))
    modes=[]
    for i in top:
        if not modes or np.linalg.norm(xy[modes]-xy[i],axis=1).min()>1:modes.append(int(i))
    return dict(pose=p.tolist(),xy_error_m=d,yaw_error_deg=angle,xy_ok=bool(d<=1),joint_ok=bool(d<=1 and angle<=30),
        separated_gap=float(z[idx//36]-z[~around].max()) if (~around).any() else 0.,
        top50_spread_m=float(np.sqrt(np.mean(np.sum((xy[top]-xy[top].mean(0))**2,axis=1)))),top50_modes=len(modes),
        xy_entropy=entropy(prob),peak_mass_1m=float(prob[around].sum()),yaw_entropy_at_map=entropy(yawprob),
        gt_joint_mass_t1=float(full[good].sum()),gt_xy_mass_t1=float(full[gtxy].sum()),
        gt_best_pose_rank=int((score>bestgt).sum()+1),gt_best_pose_rank_fraction=float((score>bestgt).mean()),
        gt_best_score_deficit=float(score.max()-bestgt)),idx


def analyze_scene(task):
    root,ssd,out,report,ref,selected=task;root,ssd,out,report=map(Path,(root,ssd,out,report))
    if ref['split']!='train':raise ValueError('TRAIN only')
    sid=ref['scene_id']; dest=out/'scenes'/sid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    source=resolve(ref,ssd);assert sha(source/'receipt.json')==ref['receipt_sha256']
    for name,digest in read(source/'receipt.json')['files'].items():assert sha(source/name)==digest
    scores=np.load(source/'scores.npy',mmap_mode='r');xy=np.load(source/'xy.npy');truth=np.load(source/'truth.npy')
    aq=root/AUDIT/'scenes'/sid; ar=read(aq/'complete.json')
    assert sha(aq/'queries.json')==ar['queries_sha256'] and sha(aq/'arrays.npz')==ar['arrays_sha256']
    queries=read(aq/'queries.json');lut={q['sample_index']:i for i,q in enumerate(queries)}
    with np.load(aq/'arrays.npz') as extra:extra=extra['extra'].copy()
    enames=ar['feature_names']
    meta=read(root/'2DFP'/sid/'floorplan_metadata.json');free=np.asarray(Image.open(root/'2DFP'/sid/'map.png').convert('L'))>127
    clearance=distance_transform_edt(free)*meta['map_spacing_m']
    shutil.copy2(root/'2DFP'/sid/'map.png',report/'maps'/(sid+'.png'))
    result=[]
    for selected_row in selected:
        i=lut[selected_row['sample_index']]; q=queries[i]; gt=truth[i]; v,a=np.asarray(scores[i]); w=selected_row['methods']['joint_success_oracle']['weight']
        assert np.allclose(gt,selected_row['truth'],atol=1e-9)
        fields=[v,a,.9*v+.1*a,(1-w)*v+w*a]; methods={}; ids={}; layers=[]
        for name,field in zip(METHODS,fields):
            methods[name],ids[name]=modality_stats(field,xy,gt)
        for name,oldname in [('fixed01','fixed_01'),('oracle','joint_success_oracle')]:
            np.testing.assert_allclose(methods[name]['pose'],selected_row['methods'][oldname]['pose'],atol=1e-8,rtol=0)
        assert methods['oracle']['joint_ok'] and not methods['fixed01']['joint_ok']
        vo,ao=methods['vision']['joint_ok'],methods['acoustic']['joint_ok']
        combination='both_right' if vo and ao else 'vision_only' if vo else 'acoustic_only' if ao else 'both_wrong'
        vx,ax=methods['vision']['xy_ok'],methods['acoustic']['xy_ok']
        xycomb='both_right' if vx and ax else 'vision_only' if vx else 'acoustic_only' if ax else 'both_wrong'
        support=success_support(q['success']['joint']); diagonal=float(np.linalg.norm(np.ptp(xy,axis=0)))
        # Three display layers: XY max score; yaw-marginal posterior T1/T.25.
        for field in fields:layers.append(field.max(1))
        for temp in (1.,.25):
            for field in fields:layers.append(softmax((field/temp).ravel()).reshape(field.shape).sum(1))
        name=sid+'_'+str(q['sample_index']); heat='heatmaps/'+name+'.bin';np.asarray(layers,dtype='<f4').tofile(report/heat)
        rgb='rgb/'+name+'.png';shutil.copy2(ssd/q['image_path'],report/rgb)
        def wall(p):
            c=int(round((p[0]-meta['world_bounds_m']['x_min'])/meta['map_spacing_m']));r=int(round((p[1]-meta['world_bounds_m']['z_min'])/meta['map_spacing_m']))
            return float(clearance[np.clip(r,0,clearance.shape[0]-1),np.clip(c,0,clearance.shape[1]-1)])
        ev={}
        for label,j in [('vision_peak',ids['vision']),('acoustic_peak',ids['acoustic'])]:
            oi=ids['oracle']; dv=float(v.ravel()[oi]-v.ravel()[j]);da=float(a.ravel()[oi]-a.ravel()[j])
            ev[label]=dict(vision_delta=dv,acoustic_delta=da,weighted_vision=(1-w)*dv,weighted_acoustic=w*da,fused_delta=(1-w)*dv+w*da)
        row=dict(scene_id=sid,split='train',sample_index=q['sample_index'],pose_id=q['pose_id'],gt=gt.tolist(),
            acoustic_weight=w,vision_weight=1-w,weight_group=group_weight(w),combination=combination,xy_combination=xycomb,
            methods=methods,success_intervals=q['success']['joint'],success_width=measure(q['success']['joint']),
            min_success_weight=min(lo for lo,hi in q['success']['joint']),max_success_weight=max(hi for lo,hi in q['success']['joint']),
            success_support=support,requires_high=bool(support['high'] and not support['low'] and not support['medium']),
            requires_low=bool(support['low'] and not support['medium'] and not support['high']),
            va_map_distance_m=float(np.linalg.norm(np.array(methods['vision']['pose'])[:2]-np.array(methods['acoustic']['pose'])[:2])),
            va_map_yaw_difference_deg=float(yawdiff(methods['vision']['pose'][2],methods['acoustic']['pose'][2])),
            gt_wall_clearance_m=wall(gt[:2]),map_diagonal_m=diagonal,
            vision_map_wall_clearance_m=wall(methods['vision']['pose'][:2]),acoustic_map_wall_clearance_m=wall(methods['acoustic']['pose'][:2]),
            depth_mean=float(extra[i,enames.index('depth_mean')]),depth_cv=float(extra[i,enames.index('depth_cv')]),
            vision_ray_mae_m=float(extra[i,enames.index('vision_ray_mae_m')]),
            acoustic_ray_mae_m=float(extra[i,enames.index('acoustic_ray_mae_m')]),evidence=ev,image=rgb,heatmap=heat)
        result.append(row)
    atomic_json(dest/'records.json',result)
    complete=dict(scene_id=sid,split='train',queries=len(result),records_sha256=sha(dest/'records.json'),test_accessed=False)
    atomic_json(dest/'complete.json',complete)
    return complete


def median(rows,fun):return float(np.median([fun(r) for r in rows])) if rows else None


def summary(rows):
    group={}
    for name in BINS:
        rr=[r for r in rows if r['weight_group']==name];n=len(rr)
        group[name]=dict(queries=n,scenes=len({r['scene_id'] for r in rr}),physical_positions=len({(r['scene_id'],r['pose_id']) for r in rr}),
            median_weight=median(rr,lambda r:r['acoustic_weight']),
            vision_joint=float(np.mean([r['methods']['vision']['joint_ok'] for r in rr])),acoustic_joint=float(np.mean([r['methods']['acoustic']['joint_ok'] for r in rr])),
            vision_xy=float(np.mean([r['methods']['vision']['xy_ok'] for r in rr])),acoustic_xy=float(np.mean([r['methods']['acoustic']['xy_ok'] for r in rr])),
            combinations={k:sum(r['combination']==k for r in rr) for k in ('vision_only','acoustic_only','both_wrong','both_right')},
            xy_combinations={k:sum(r['xy_combination']==k for r in rr) for k in ('vision_only','acoustic_only','both_wrong','both_right')},
            successful_in_bins={g:sum(r['success_support'][g] for r in rr) for g in BINS},
            requires_high=sum(r['requires_high'] for r in rr),requires_low=sum(r['requires_low'] for r in rr),
            medians={k:median(rr,lambda r,k=k:r[k]) for k in ('depth_mean','depth_cv','gt_wall_clearance_m','va_map_distance_m','va_map_yaw_difference_deg','success_width','min_success_weight','vision_ray_mae_m','acoustic_ray_mae_m')},
            modality_medians={m:{k:median(rr,lambda r,m=m,k=k:r['methods'][m][k]) for k in ('xy_error_m','yaw_error_deg','separated_gap','top50_spread_m','top50_modes','xy_entropy','peak_mass_1m','yaw_entropy_at_map','gt_joint_mass_t1','gt_best_pose_rank','gt_best_pose_rank_fraction')} for m in ('vision','acoustic')})
    return group


def representatives(rows):
    names=['depth_mean','depth_cv','va_map_distance_m','va_map_yaw_difference_deg','gt_wall_clearance_m']
    x=np.array([[r[k] for k in names]+[r['methods'][m][k] for m in ('vision','acoustic') for k in ('xy_error_m','yaw_error_deg','separated_gap','top50_spread_m')] for r in rows])
    scale=np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0),1e-5);x=x/scale
    picks=[]
    for g in BINS:
        ix=np.array([i for i,r in enumerate(rows) if r['weight_group']==g]);centre=np.median(x[ix],axis=0)
        order=ix[np.argsort(np.sum((x[ix]-centre)**2,axis=1))];seen=set();selected=[]
        for i in order:
            if rows[i]['scene_id'] not in seen:
                selected.append(int(i));seen.add(rows[i]['scene_id'])
            if len(selected)==3:break
        for c in ('vision_only','acoustic_only','both_wrong','both_right'):
            eligible=[int(i) for i in order if rows[i]['combination']==c]
            if eligible and not any(rows[i]['combination']==c for i in selected):selected.append(eligible[0])
        for i in selected:picks.append(dict(index=i,group=g,scene_id=rows[i]['scene_id'],sample_index=rows[i]['sample_index'],combination=rows[i]['combination']))
    return picks


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    for d in ('maps','rgb','heatmaps'):(report/d).mkdir(exist_ok=True)
    oracle=read(root/ORACLE/'verified_records.json');assert len(oracle)==6387 and {r['split'] for r in oracle}=={'train'}
    selected=[]
    for r in oracle:
        pp=np.array(r['methods']['joint_success_oracle']['pose']);gt=np.array(r['truth'])
        if not r['baseline_joint_success'] and np.linalg.norm(pp[:2]-gt[:2])<=1 and yawdiff(pp[2],gt[2])<=30:selected.append(r)
    assert len(selected)==967
    refs=[r for r in read(root/SOURCE/'scores_ready.json')['entries'] if r['split']=='train'];assert len(refs)==12
    config=dict(version='oracle_weight_groups_v1',oracle_sha256=sha(root/ORACLE/'verified_records.json'),cohort='GT oracle JOINT repair vs fixedw.1 ONLY',
        train_total=6387,baseline_correct=4493,unrecoverable=927,repair_queries=967,bins=BINS,
        chosen_weight_notice='Oracle chosen weight is not unique; whole positive-width success intervals inspected. GT conditional cohort, not deployable reliability labels.',
        layer_order=[m+'_xymax' for m in METHODS]+[m+'_marginal_T'+str(t) for t in (1.,.25) for m in METHODS],
        display='XY max score OR XY probability marginal over yaw, common temperature per view; both are pseudo-likelihoods, not calibrated confidence',
        representatives='IQR-scaled group-centre examples with up to3 different scenes, plus missing joint-outcome combinations; not performance-selected',
        branch_notice='Frozen Train predictions are in-sample, not out-of-fold. Observational GT-conditioned analysis, no causal/generalization claim.',test_accessed=False,val_accessed=False)
    immutable_json(out/'config.json',config)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[(str(root),str(ssd),str(out),str(report),ref,[r for r in selected if r['scene_id']==ref['scene_id']]) for ref in refs]
        for k,f in enumerate(as_completed([pool.submit(analyze_scene,t) for t in tasks]),1):
            info=f.result();atomic_json(out/'status.json',dict(status='analyzing',done=k,total=12,last=info['scene_id'],test_accessed=False));print('SCENE',info,flush=True)
    rows=[r for ref in refs for r in read(out/'scenes'/ref['scene_id']/'records.json')];assert len(rows)==967
    banks={}
    for ref in refs:
        sid=ref['scene_id'];banks[sid]=dict(xy=np.load(resolve(ref,ssd)/'xy.npy').tolist(),metadata=read(root/'2DFP'/sid/'floorplan_metadata.json'))
    primary=summary(rows)
    # Data-dependent tertiles are sensitivity ONLY, not primary group selection.
    cuts=np.quantile([r['acoustic_weight'] for r in rows],[1/3,2/3]);alt=[]
    for r in rows:
        q=dict(r);w=q['acoustic_weight'];q['weight_group']='low' if w<cuts[0] else 'medium' if w<cuts[1] else 'high';alt.append(q)
    keys=['vision_joint','acoustic_joint']
    scene_effects={}
    for feature,fun in [('vision_peak_gap',lambda r:r['methods']['vision']['separated_gap']),('acoustic_spread_m',lambda r:r['methods']['acoustic']['top50_spread_m']),('depth_mean',lambda r:r['depth_mean']),('gt_wall_clearance_m',lambda r:r['gt_wall_clearance_m'])]:
        comparisons=[]
        for sid in banks:
            lo=[r for r in rows if r['scene_id']==sid and r['weight_group']=='low'];hi=[r for r in rows if r['scene_id']==sid and r['weight_group']=='high']
            if len(lo)>=5 and len(hi)>=5:comparisons.append(dict(scene_id=sid,low_n=len(lo),high_n=len(hi),high_minus_low=median(hi,fun)-median(lo,fun)))
        scene_effects[feature]=comparisons
    result=dict(status='complete',config=config,groups=primary,tertile_sensitivity=dict(cuts=cuts.tolist(),groups=summary(alt)),scene_adjusted_descriptive=scene_effects,
        per_scene={sid:summary([r for r in rows if r['scene_id']==sid]) for sid in banks if all(any(r['scene_id']==sid and r['weight_group']==g for r in rows) for g in BINS)},
        representatives=representatives(rows),test_accessed=False,val_accessed=False)
    atomic_json(out/'summary.json',result);atomic_json(out/'records.json',rows)
    flat=[]
    for r in rows:
        d={k:r[k] for k in ('scene_id','sample_index','pose_id','acoustic_weight','vision_weight','weight_group','combination','xy_combination','success_width','min_success_weight','max_success_weight','requires_high','requires_low','va_map_distance_m','gt_wall_clearance_m','depth_mean','depth_cv')}
        for m in METHODS:
            d.update({m+'_'+k:r['methods'][m][k] for k in ('xy_error_m','yaw_error_deg','joint_ok','separated_gap','top50_spread_m','gt_joint_mass_t1','gt_best_pose_rank')})
        flat.append(d)
    with (out/'queries.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    for name in ('summary.json','records.json','queries.csv'):shutil.copy2(out/name,report/name)
    atomic_json(report/'banks.json',banks)
    shutil.copy2(Path(__file__).with_name('oracle_weight_groups_report.html'),report/'index.html')
    # Static scientific summary, no external plotting service.
    fig,ax=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True);groups=list(BINS);x=np.arange(3)
    ax[0].bar(x-.18,[primary[g]['vision_joint']*100 for g in groups],.36,label='Vision MAP',color='#377eb8')
    ax[0].bar(x+.18,[primary[g]['acoustic_joint']*100 for g in groups],.36,label='Acoustic MAP',color='#ed9832')
    ax[0].set(xticks=x,xticklabels=[f'{g}\nn={primary[g]["queries"]}' for g in groups],ylabel='Joint success (%)',ylim=(0,100),title='Individual MAPs within oracle-repaired queries');ax[0].legend()
    for g,color in zip(groups,['#477ac4','#54a27b','#d9933c']):
        w=[r['acoustic_weight'] for r in rows if r['weight_group']==g];ax[1].hist(w,bins=np.linspace(0,1,41),histtype='stepfilled',alpha=.6,label=g,color=color)
    ax[1].set(xlabel='Chosen acoustic coefficient',ylabel='Queries',title='Chosen oracle weight (not a unique GT label)');ax[1].legend()
    fig.savefig(out/'group_summary.png',dpi=160);plt.close(fig);shutil.copy2(out/'group_summary.png',report/'group_summary.png')
    verification=dict(status='passed',train_queries=6387,analyzed_repaired_queries=967,group_counts={g:primary[g]['queries'] for g in BINS},
        fixed_and_oracle_poses_reproduced=967*2,scene_ids=[r['scene_id'] for r in refs],test_accessed=False,val_accessed=False,
        source_code_sha256=sha(Path(__file__)),config_sha256=sha(out/'config.json'))
    atomic_json(out/'verification.json',verification);atomic_json(report/'verification.json',verification)
    atomic_json(out/'status.json',dict(status='complete',test_accessed=False));print('COMPLETE',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--workers',type=int,default=6);main(p.parse_args())
