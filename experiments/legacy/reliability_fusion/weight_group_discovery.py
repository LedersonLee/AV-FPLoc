"""Exploratory TRAIN-only, GT-free feature discovery for feasible weight sets.

GT is confined to labels/evaluation and never passed to novel_features. Probe
classifiers are scene-held-out diagnostics, NOT a new deployed fusion gate.
All Train queries, including fixed-.1 successes, are retained. A chosen oracle
weight is only a secondary, explicitly nonunique diagnostic label.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.special import softmax
from scipy.stats import rankdata

from acoustic_branch.contracts import atomic_json, immutable_json, sha
from .input_supervision_audit import SOURCE, REL as AUDIT, CORE_NAMES, read, resolve, pose, yawdiff, measure, winner_at
from .oracle_weight_groups import success_support, group_weight

REL = 'ReliabilityFusion/analysis/weight_group_discovery_v1'
ORACLE = 'ReliabilityFusion/pilots/independent_pose_risk_v1'
GROUP_SOURCE = 'ReliabilityFusion/analysis/oracle_weight_groups_v1'
SEED = 20261021
MASK_NAMES = ['none', 'low', 'medium', 'low+medium', 'high', 'low+high', 'medium+high', 'low+medium+high']


def mask_of(intervals):
    support = success_support(intervals)
    return sum((1 << i) for i, g in enumerate(('low', 'medium', 'high')) if support[g])


def basin_features(parts,xy):
    """Response around predicted fixed-.1 MAP, NOT the GT neighbourhood."""
    pp=np.array([pose(int(idx),xy)for lo,hi,idx in parts]);base=pose(winner_at(parts,.1),xy)
    names=[];vals=[];width=np.array([hi-lo for lo,hi,idx in parts])
    def add(n,v):names.append('basin_'+n);vals.append(float(v))
    for label,good in [('xy',np.linalg.norm(pp[:,:2]-base[:2],axis=1)<=1.),('joint',(np.linalg.norm(pp[:,:2]-base[:2],axis=1)<=1.)&(yawdiff(pp[:,2],base[2])<=30))]:
        from .input_supervision_audit import merge_intervals
        intervals=merge_intervals([[lo,hi]for(lo,hi,_),ok in zip(parts,good)if ok]);lo,hi=next((lo,hi)for lo,hi in intervals if lo-1e-12<=.1<=hi+1e-12)
        add(label+'_left_stability',.1-lo);add(label+'_right_stability',hi-.1)
        add(label+'_total_weight_support',width[good].sum());add(label+'_reentries',len(intervals)-1)
    dp=np.linalg.norm(np.diff(pp[:,:2],axis=0),axis=1);da=yawdiff(pp[1:,2],pp[:-1,2]);bounds=np.array([lo for lo,hi,idx in parts][1:]);active=(bounds>.1)&(bounds<.5)
    add('large_xy_jumps_to05',np.sum((dp>1)&active));add('large_yaw_jumps_to05',np.sum((da>30)&active));add('large_joint_jumps_to05',np.sum(((dp>1)|(da>30))&active))
    add('small_grid_jumps_to05',np.sum((dp<=1)&(da<=30)&active))
    centres=[];masses=[]
    for p,w in zip(pp,width):
        match=[j for j,c in enumerate(centres)if np.linalg.norm(c[:2]-p[:2])<=1 and yawdiff(c[2],p[2])<=30]
        if match:masses[match[0]]+=w
        else:centres.append(p);masses.append(w)
    add('joint_mode_count',len(centres));add('joint_mode_max_weight_mass',max(masses));add('joint_mode_weight_entropy',entropy(np.array(masses))/np.log(max(2,len(masses))))
    return np.array(vals),names


def entropy(p):
    return float(-np.sum(p * np.log(np.maximum(p, 1e-300))))


def normalize(s):
    return (s - s.mean()) / max(float(s.std()), 1e-12)


def novel_features(v, a, xy, parts, geometry, rays, rgb):
    """Observed scores, predicted-MAP geometry, and RGB only; no GT/labels.

    No oracle/future chosen weight, absolute pose, scene ID, RIR at GT, or
    Laplace scale is exposed to feature construction. DESDF is object-free.
    """
    names, vals, families = [], [], []

    def add(name, value, family):
        names.append(name); vals.append(float(value)); families.append(family)

    fields = (v, a); ids = [int(x.argmax()) for x in fields]
    baseid = winner_at(parts, .1); ids3 = [*ids, baseid]
    diag = max(float(np.linalg.norm(np.ptp(xy, axis=0))), 1.)
    dv = float(v.ravel()[ids[0]] - v.ravel()[ids[1]])
    da = float(a.ravel()[ids[1]] - a.ravel()[ids[0]])
    add('map_pair_crossing_weight', dv / (dv + da) if dv + da > 1e-12 else .5, 'competition')
    add('map_pair_total_conflict', dv + da, 'competition')
    add('map_pair_log_deficit_ratio', np.log((dv + 1e-8) / (da + 1e-8)), 'competition')
    add('fixed01_vision_deficit', v.max() - v.ravel()[baseid], 'competition')
    add('fixed01_acoustic_deficit', a.max() - a.ravel()[baseid], 'competition')
    add('fixed01_acoustic_advantage_over_vmap', a.ravel()[baseid] - a.ravel()[ids[0]], 'competition')
    add('fixed01_vision_advantage_over_amap', v.ravel()[baseid] - v.ravel()[ids[1]], 'competition')
    add('pose_score_pearson', np.corrcoef(v.ravel(), a.ravel())[0, 1] if v.std()*a.std()>1e-15 else 0, 'agreement')
    spatial = [x.max(1) for x in fields]
    vv, aa = [rankdata(x) for x in spatial]
    add('xy_score_spearman', np.corrcoef(vv, aa)[0, 1] if vv.std()*aa.std()>1e-15 else 0, 'agreement')
    # Shared temperature; do not independently rescale branch scores here.
    probs = [softmax((x / .25).ravel()).reshape(x.shape) for x in fields]
    pp, qq = probs; mix = (pp + qq) / 2
    add('pose_hellinger_affinity_t025', np.sqrt(pp*qq).sum(), 'agreement')
    add('pose_js_divergence_t025', entropy(mix)-(entropy(pp)+entropy(qq))/2, 'agreement')
    xp, xq = pp.sum(1), qq.sum(1)
    add('xy_hellinger_affinity_t025', np.sqrt(xp*xq).sum(), 'agreement')
    add('xy_js_divergence_t025', entropy((xp+xq)/2)-(entropy(xp)+entropy(xq))/2, 'agreement')
    yp, yq = pp.sum(0), qq.sum(0)
    add('yaw_hellinger_affinity_t025', np.sqrt(yp*yq).sum(), 'orientation')
    # Rank support captures more than agreement at a single MAP.
    for frac in (.001, .01):
        n = max(1, int(np.ceil(v.size*frac)))
        top = [np.argpartition(-x.ravel(), n-1)[:n] for x in fields]
        add(f'top{frac:g}_pose_overlap', len(np.intersect1d(*top))/n, 'agreement')
        for k, label in enumerate(('vision', 'acoustic')):
            other = top[1-k]
            add(f'{label}_support_other_top{frac:g}', (fields[k].max()-fields[k].ravel()[other]).mean(), 'competition')
    for k, label in enumerate(('vision', 'acoustic')):
        s, p = fields[k], probs[k]; marg = p.sum(1); yaw = p.sum(0)
        add(label+'_score_std', s.std(), 'shape')
        add(label+'_score_tail_999', s.max()-np.quantile(s, .999), 'shape')
        add(label+'_pose_entropy_t025', entropy(p)/np.log(p.size), 'shape')
        add(label+'_xy_marginal_entropy_t025', entropy(marg)/np.log(len(marg)), 'shape')
        add(label+'_conditional_yaw_entropy_t025', (entropy(p)-entropy(marg))/np.log(36), 'orientation')
        add(label+'_global_yaw_entropy_t025', entropy(yaw)/np.log(36), 'orientation')
        angle = np.arange(36)*2*np.pi/36
        for harmonic in (1, 2, 4):
            add(label+f'_yaw_resultant_h{harmonic}', abs(np.sum(yaw*np.exp(1j*harmonic*angle))), 'orientation')
        owncell, othercell = ids[k]//36, ids[1-k]//36
        other_yaw = ids[1-k]%36; yawrow = softmax(s[othercell]/.25)
        add(label+'_yaw_mass_at_other_heading', yawrow[yawdiff(np.arange(36)*10, other_yaw*10)<=30].sum(), 'orientation')
        add(label+'_own_yaw_gap90', s[owncell].max()-s[owncell][yawdiff(np.arange(36)*10, ids[k]%36*10)>=90].max(), 'orientation')
        topxy = np.argpartition(-spatial[k], min(49, len(xy)-1))[:min(50,len(xy))]
        xy_top = xy[topxy]; eig = np.linalg.eigvalsh(np.cov(xy_top.T))
        add(label+'_top50_anisotropy', eig[-1]/max(float(eig.sum()),1e-12), 'shape')
        topyaw = s[topxy].argmax(1)
        add(label+'_top50_yaw_agreement', np.mean(yawdiff(topyaw*10, ids[k]%36*10)<=30), 'orientation')
        # Multi-scale concentration is insensitive to scene size in metres.
        d = np.linalg.norm(xy-xy[owncell],axis=1)
        for radius in (.5, 2., 4.):
            add(label+f'_mass_radius{radius:g}m_t025', marg[d<=radius].sum(), 'shape')
    trace = [pose(int(idx),xy) for lo,hi,idx in parts]
    widths = np.array([hi-lo for lo,hi,idx in parts])
    add('envelope_winner_count', len(parts), 'dynamics')
    add('envelope_largest_stable_width', widths.max(), 'dynamics')
    add('envelope_width_entropy', entropy(widths)/np.log(max(2,len(widths))), 'dynamics')
    dp = np.diff(np.asarray(trace)[:,:2],axis=0)
    add('envelope_xy_path_over_diagonal', np.linalg.norm(dp,axis=1).sum()/diag, 'dynamics')
    add('envelope_yaw_path_turns', yawdiff(np.array(trace)[1:,2],np.array(trace)[:-1,2]).sum()/360., 'dynamics')
    for w in (.025,.075,.125,.25,.75):
        pi=pose(winner_at(parts,w),xy); p0=pose(baseid,xy)
        add(f'probe{w:g}_fixed_distance', np.linalg.norm(pi[:2]-p0[:2])/diag, 'dynamics')
        add(f'probe{w:g}_fixed_yaw', yawdiff(pi[2],p0[2])/180., 'dynamics')
    meta, geom = geometry
    def at(grid,p):
        c=int(round((p[0]-meta['world_bounds_m']['x_min'])/meta['map_spacing_m']))
        r=int(round((p[1]-meta['world_bounds_m']['z_min'])/meta['map_spacing_m']))
        return grid[np.clip(r,0,grid.shape[0]-1),np.clip(c,0,grid.shape[1]-1)]
    for label,idx in zip(('vision','acoustic','fixed01'),ids3):
        p=pose(idx,xy)
        for name,grid in geom.items():add(label+'_'+name,at(grid,p),'geometry')
        rr=rays[idx//36]; forward=rr[(idx%36+np.arange(-5,6))%36]
        add(label+'_fp_forward_mean',forward.mean(),'geometry')
        add(label+'_fp_forward_cv',forward.std()/max(float(forward.mean()),.1),'geometry')
        add(label+'_fp_near_fraction',np.mean(rr<=1.),'geometry')
        add(label+'_fp_shortest_ray',rr.min(),'geometry')
        for harmonic in (1,2):
            add(label+f'_fp_ray_h{harmonic}',abs(np.sum(rr*np.exp(1j*harmonic*np.arange(36)*2*np.pi/36)))/max(float(rr.sum()),1e-8),'geometry')
    # Cheap observable RGB descriptors, not inferred semantic/occlusion labels.
    im=np.asarray(rgb.resize((64,48)).convert('RGB'),float)/255.
    lum=im@np.array([.2126,.7152,.0722]); hist=np.histogram(lum,bins=32,range=(0,1))[0]/lum.size
    add('rgb_luminance_mean',lum.mean(),'rgb');add('rgb_luminance_std',lum.std(),'rgb')
    add('rgb_entropy',entropy(hist)/np.log(32),'rgb')
    add('rgb_dark_fraction',np.mean(lum<.05),'rgb');add('rgb_bright_fraction',np.mean(lum>.95),'rgb')
    add('rgb_color_spread',im.std((0,1)).mean(),'rgb')
    dx=np.abs(np.diff(lum,axis=1)).mean();dy=np.abs(np.diff(lum,axis=0)).mean()
    add('rgb_gradient_mean',dx+dy,'rgb');add('rgb_edge_orientation',dx/max(dx+dy,1e-8),'rgb')
    add('rgb_mirror_difference',np.abs(lum-lum[:,::-1]).mean(),'rgb')
    add('rgb_vertical_brightness_difference',lum[:24].mean()-lum[24:].mean(),'rgb')
    out=np.array(vals,float)
    if not np.isfinite(out).all():raise ValueError('Nonfinite GT-free features')
    return out,names,families


def extract_scene(task):
    root,ssd,out,ref=task;root,ssd,out=map(Path,(root,ssd,out));sid=ref['scene_id']
    assert ref['split']=='train';dest=out/'scenes'/sid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        receipt=read(dest/'complete.json');assert sha(dest/'features.npz')==receipt['features_sha256'];return receipt
    source=resolve(ref,ssd);assert sha(source/'receipt.json')==ref['receipt_sha256']
    receipt=read(source/'receipt.json')
    for name,digest in receipt['files'].items():assert sha(source/name)==digest
    scores=np.load(source/'scores.npy',mmap_mode='r');xy=np.load(source/'xy.npy');truth=np.load(source/'truth.npy')
    aq=root/AUDIT/'scenes'/sid; ar=read(aq/'complete.json')
    assert sha(aq/'arrays.npz')==ar['arrays_sha256'] and sha(aq/'queries.json')==ar['queries_sha256']
    qs=read(aq/'queries.json');assert all(q['split']=='train' for q in qs)
    with np.load(aq/'arrays.npz') as b:old=np.c_[b['core'],b['extra']];np.testing.assert_allclose(b['truth'],truth)
    meta=read(root/'2DFP'/sid/'floorplan_metadata.json');free=np.array(Image.open(root/'2DFP'/sid/'map.png').convert('L'))>127
    spacing=meta['map_spacing_m'];geom={'wall_clearance':distance_transform_edt(free)*spacing}
    for radius in (.5,1.,2.):geom[f'local_free_square{radius:g}m']=uniform_filter(free.astype(float),size=2*int(round(radius/spacing))+1,mode='constant')
    payload=np.load(ssd/'UnLoc/structural_targets'/sid/'desdf.npy',allow_pickle=True).item()
    assert payload['desdf_spacing_m']==.1 and payload['max_range_m']==10
    origin=np.array([payload['origin_x_m'],payload['origin_z_min_m']]);ij=np.rint((xy-origin)/.1).astype(int)
    np.testing.assert_allclose(origin+ij*.1,xy,atol=1e-6,rtol=0);rays=payload['desdf'][ij[:,1],ij[:,0]]
    features=[];diagnostics=[]
    for i,q in enumerate(qs):
        v,a=np.asarray(scores[i]);vals,names,families=novel_features(v,a,xy,q['parts'],(meta,geom),rays,Image.open(ssd/q['image_path']))
        features.append(vals);pp=[pose(int(f.argmax()),xy) for f in (v,a,.9*v+.1*a)]
        diagnostics.append([dict(pose=p.tolist(),xy_error_m=float(np.linalg.norm(p[:2]-truth[i,:2])),yaw_error_deg=float(yawdiff(p[2],truth[i,2])))for p in pp])
        if i%100==0:atomic_json(dest/'status.json',dict(done=i,total=len(qs),split='train'));print(sid,i,len(qs),flush=True)
    names=CORE_NAMES+ar['feature_names']+names;families=['core14']*14+['prior_'+f for f in ar['feature_families']]+families
    x=np.c_[old,np.array(features)];assert len(names)==x.shape[1] and len(set(names))==len(names)
    np.savez_compressed(dest/'features.npz',x=x,samples=np.array([q['sample_index']for q in qs]))
    atomic_json(dest/'diagnostics.json',diagnostics)
    result=dict(scene_id=sid,split='train',queries=len(qs),names=names,families=families,features_sha256=sha(dest/'features.npz'),diagnostics_sha256=sha(dest/'diagnostics.json'),source_receipt_sha256=ref['receipt_sha256'])
    atomic_json(dest/'complete.json',result);print('COMPLETE',sid,flush=True);return result


def metric(y,p):
    from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix,roc_auc_score
    labels=np.arange(p.shape[1]);pred=p.argmax(1)
    return dict(n=len(y),balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),accuracy=float(np.mean(y==pred)),confusion=confusion_matrix(y,pred,labels=labels).tolist(),macro_auc=float(roc_auc_score(np.eye(len(labels))[y],p,average='macro')))


def get_tasks(rows):
    mask=np.array([r['mask']for r in rows]);old=np.array([r['legacy_group']for r in rows]);base=np.array([r['baseline_success']for r in rows])
    return {
      'legacy_repaired3':(np.flatnonzero(old>=0),old[old>=0],['low','medium','high']),
      'exclusive3':(np.flatnonzero(np.isin(mask,[1,2,4])),np.array([{1:0,2:1,4:2}[m]for m in mask if m in(1,2,4)]),['only_low','only_medium','only_high']),
      'all_feasible8':(np.arange(len(rows)),mask,MASK_NAMES),
      'all_fixed01_success':(np.arange(len(rows)),base.astype(int),['fixed01_fail','fixed01_success']),
    }


def variants(names,families):
    fam=np.array(families);result={'core14':np.arange(14),'existing58':np.arange(58),'new_only':np.arange(58,len(names)),'all':np.arange(len(names))}
    for family in ('competition','agreement','orientation','shape','dynamics','geometry','rgb'):
        result['only_'+family]=np.flatnonzero(fam==family)
        result['existing58+'+family]=np.r_[np.arange(58),np.flatnonzero(fam==family)]
    return result


def probe(task):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from threadpoolctl import threadpool_limits
    folder,taskname,modelkind=task;folder=Path(folder);rows=read(folder/'query_index.json');spec=read(folder/'features.json')
    ix,y,labels=get_tasks(rows)[taskname];x=np.load(folder/'features.npy',mmap_mode='r')[ix].copy();groups=np.array([rows[i]['scene_id']for i in ix]);nclass=len(labels)
    dest=folder/'probes'/f'{taskname}_{modelkind}.json';dest.parent.mkdir(exist_ok=True)
    if dest.exists():return read(dest)
    results=[];probabilities={};perms=[];rng=np.random.default_rng(SEED)
    with threadpool_limits(limits=1):
      for name,cols in variants(spec['names'],spec['families']).items():
        oof=np.zeros((len(y),nclass));foldmetrics=[]
        for sid in np.unique(groups):
            a=np.flatnonzero(groups!=sid);b=np.flatnonzero(groups==sid)
            assert not set(groups[a])&set(groups[b]);assert len(np.unique(y[a]))==nclass
            scale=StandardScaler().fit(x[a][:,cols]);xa=scale.transform(x[a][:,cols]);xb=scale.transform(x[b][:,cols])
            if modelkind=='linear':m=LogisticRegression(C=.1,max_iter=1500,class_weight='balanced',random_state=SEED)
            else:m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=25,l2_regularization=2.,early_stopping=False,random_state=SEED)
            counts=np.bincount(y[a],minlength=nclass);weights=len(a)/(nclass*counts[y[a]])
            if modelkind=='linear':m.fit(xa,y[a])
            else:m.fit(xa,y[a],sample_weight=weights)
            oof[b]=m.predict_proba(xb);foldmetrics.append(dict(scene_id=str(sid),n=len(b),balanced_accuracy=float(balanced_accuracy_score(y[b],oof[b].argmax(1)))))
            if name=='all' and modelkind=='tree':
                base=float(balanced_accuracy_score(y[b],oof[b].argmax(1)))
                for family in sorted(set(spec['families'])):
                    sub=np.flatnonzero(np.array(spec['families'])==family);deltas=[]
                    for _ in range(5):
                        xp=xb.copy();xp[:,sub]=xb[rng.permutation(len(b))][:,sub]
                        deltas.append(base-float(balanced_accuracy_score(y[b],m.predict(xp))))
                    perms.append(dict(scene_id=str(sid),family=family,n=len(b),delta_balanced_accuracy=float(np.mean(deltas))))
        results.append(dict(variant=name,input_count=len(cols),**metric(y,oof),scene_folds=foldmetrics));probabilities[name]=oof
        print('PROBE',taskname,modelkind,name,results[-1]['balanced_accuracy'],flush=True)
    np.savez_compressed(dest.with_suffix('.npz'),y=y,global_ids=ix,**probabilities)
    value=dict(task=taskname,model=modelkind,labels=labels,results=results,permutations=perms,folds=12,validation='Leave ONE Train scene out, no Val/Test; encoders frozen and already trained on these scenes')
    atomic_json(dest,value);return value


def single_features(task):
    """Depth-2 univariate probes, fixed bins fit ONLY on other Train scenes.

    Ranking many OOF results is exploratory, not an unbiased selected-feature
    estimate. A separate nested-selector baseline avoids that selection bias.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score,balanced_accuracy_score
    from threadpoolctl import threadpool_limits
    folder,taskname=task;folder=Path(folder);dest=folder/'single'/f'{taskname}.json';dest.parent.mkdir(exist_ok=True)
    if dest.exists():return read(dest)
    rows=read(folder/'query_index.json');spec=read(folder/'features.json');ix,y,labels=get_tasks(rows)[taskname]
    x=np.load(folder/'features.npy',mmap_mode='r')[ix].copy();groups=np.array([rows[i]['scene_id']for i in ix]);result=[]
    with threadpool_limits(limits=1):
      for j,name in enumerate(spec['names']):
        p=np.zeros((len(ix),len(labels)))
        for sid in np.unique(groups):
            a=groups!=sid;b=~a;m=DecisionTreeClassifier(max_depth=2,min_samples_leaf=25,class_weight='balanced',random_state=SEED)
            m.fit(x[a,j:j+1],y[a]);p[b]=m.predict_proba(x[b,j:j+1])
        perclass=[float(roc_auc_score(y==k,p[:,k]))for k in range(len(labels))]
        result.append(dict(name=name,family=spec['families'][j],balanced_accuracy=float(balanced_accuracy_score(y,p.argmax(1))),auc_per_class=perclass,macro_auc=float(np.mean(perclass)),medians=[float(np.median(x[y==k,j]))for k in range(len(labels))]))
    value=dict(task=taskname,labels=labels,features=sorted(result,key=lambda r:-r['macro_auc']),notice='OOF ranking across many features is exploratory; no significance or selected-policy claim')
    atomic_json(dest,value);return value


def run(args):
    root,ssd=args.root,args.ssd;out=root/REL;out.mkdir(parents=True,exist_ok=True)
    refs=[r for r in read(root/SOURCE/'scores_ready.json')['entries']if r['split']=='train'];assert len(refs)==12
    oracle=read(root/ORACLE/'verified_records.json');assert len(oracle)==6387 and all(r['split']=='train'for r in oracle)
    immutable_json(out/'config.json',dict(seed=SEED,version='v1',cohort='ALL Train12, fixed01 successes INCLUDED',oracle_sha256=sha(root/ORACLE/'verified_records.json'),test_accessed=False,val_accessed=False,primary_labels='8 full feasible-bin masks; fixed01 success separate; exclusive3 and legacy chosen3 secondary',feature_policy='No GT, oracle weight, success interval, pose error, absolute pose, scene ID, or Laplace scale in feature construction',probe='12-fold leave-one-scene-out, fixed linear/tree hyperparameters; no deployment fusion model trained'))
    if not args.probe_only:
      with ProcessPoolExecutor(max_workers=args.workers)as pool:
        for f in as_completed([pool.submit(extract_scene,(str(root),str(ssd),str(out),ref))for ref in refs]):
            info=f.result();atomic_json(out/'status.json',dict(stage='extract',last=info['scene_id']))
    matrices=[];rows=[];lookup={(r['scene_id'],r['sample_index']):r for r in oracle}
    for ref in refs:
        sid=ref['scene_id'];dest=out/'scenes'/sid;receipt=read(dest/'complete.json')
        assert sha(dest/'features.npz')==receipt['features_sha256'] and sha(dest/'diagnostics.json')==receipt['diagnostics_sha256']
        with np.load(dest/'features.npz')as b:matrices.append(b['x'].copy());samples=b['samples'].copy()
        q=read(root/AUDIT/'scenes'/sid/'queries.json');diags=read(dest/'diagnostics.json')
        assert np.array_equal(samples,[z['sample_index']for z in q])
        for z,d in zip(q,diags):
            r=lookup[(sid,z['sample_index'])];mask=mask_of(z['success']['joint']);fixed=r['baseline_joint_success'];recover=r['success_possible']
            assert bool(mask)==recover
            old=-1 if fixed or not recover else ('low','medium','high').index(group_weight(r['methods']['joint_success_oracle']['weight']))
            rows.append(dict(scene_id=sid,split='train',sample_index=z['sample_index'],pose_id=z['pose_id'],mask=mask,mask_name=MASK_NAMES[mask],baseline_success=fixed,legacy_group=old,success_intervals=z['success']['joint'],success_width=measure(z['success']['joint']),success_components=len(z['success']['joint']),truth=z['truth'],methods=dict(zip(('vision','acoustic','fixed01'),d)),image_path=z['image_path'],oracle_weight=r['methods']['joint_success_oracle']['weight'],oracle_pose=r['methods']['joint_success_oracle']['pose']))
    x=np.concatenate(matrices);assert len(rows)==6387 and np.isfinite(x).all()
    np.save(out/'features.npy',x);atomic_json(out/'query_index.json',rows);atomic_json(out/'features.json',dict(names=receipt['names'],families=receipt['families']))
    tasks=[(str(out),name,kind)for name in get_tasks(rows)for kind in ('linear','tree')]
    with ProcessPoolExecutor(max_workers=min(args.workers,8))as pool:
        for f in as_completed([pool.submit(probe,t)for t in tasks]):
            info=f.result();atomic_json(out/'status.json',dict(stage='probe',last=info['task']+'_'+info['model']))
    with ProcessPoolExecutor(max_workers=min(args.workers,4))as pool:
        for f in as_completed([pool.submit(single_features,(str(out),name))for name in get_tasks(rows)]):f.result()
    config=read(out/'config.json');spec=read(out/'features.json');summary=[]
    for mask in range(8):
        ids=[i for i,r in enumerate(rows)if r['mask']==mask];rr=[rows[i]for i in ids]
        summary.append(dict(mask=mask,name=MASK_NAMES[mask],queries=len(rr),scenes=len(set(r['scene_id']for r in rr)),fixed01_success=sum(r['baseline_success']for r in rr),median_success_width=float(np.median([r['success_width']for r in rr])),median_success_components=float(np.median([r['success_components']for r in rr])),feature_medians=dict(zip(spec['names'],np.median(x[ids],axis=0).tolist()))))
    result=dict(config=config,groups=summary,feature_count=len(spec['names']),feature_families={f:spec['families'].count(f)for f in sorted(set(spec['families']))},probes=[read(out/'probes'/f'{t}_{m}.json')for t in get_tasks(rows)for m in ('linear','tree')],single_features=[read(out/'single'/f'{t}.json')for t in get_tasks(rows)],test_accessed=False,val_accessed=False)
    atomic_json(out/'results.json',result);atomic_json(out/'status.json',dict(stage='complete',queries=len(rows),test_accessed=False,val_accessed=False));print('COMPLETE ALL',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--probe-only',action='store_true');run(p.parse_args())
