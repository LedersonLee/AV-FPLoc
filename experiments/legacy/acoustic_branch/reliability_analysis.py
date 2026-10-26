"""Exploratory, paired test failure mechanisms; never fits a fusion policy.

Reconstruct the SAME frozen contrastive test scores from saved embeddings and
vision outputs, and require every four-way pose choice to reproduce the report.
GT diagnostics are explicitly separated from inference-available predictors.
No new acoustic simulation or checkpoint training. CPU only.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.special import softmax
from scipy.stats import rankdata

from .contracts import atomic_json,sha
from .data import read
from .evaluation import METHODS,select,standardized
from .pilot_evaluate import angular_rays,bank_windows


FEATURES={
 'v_entropy':('Vision XY entropy','inference'),
 'a_entropy':('Acoustic XY entropy','inference'),
 'v_peak_mass_1m':('Vision mass within 1m of MAP','inference'),
 'a_peak_mass_1m':('Acoustic mass within 1m of MAP','inference'),
 'v_separated_gap':('Vision gap to best peak >1m away (z)','inference'),
 'a_separated_gap':('Acoustic gap to best peak >1m away (z)','inference'),
 'v_top50_modes':('Vision top50 separated modes (>1m)','inference'),
 'a_top50_modes':('Acoustic top50 separated modes (>1m)','inference'),
 'v_top50_rms_m':('Vision top50 RMS spread (m)','inference'),
 'a_top50_rms_m':('Acoustic top50 RMS spread (m)','inference'),
 'v_yaw_entropy':('Vision yaw entropy at its MAP cell','inference'),
 'a_yaw_entropy':('Acoustic yaw entropy at its MAP cell','inference'),
 'map_disagreement_m':('Vision/acoustic MAP separation (m)','inference'),
 'map_yaw_disagreement_deg':('Vision/acoustic MAP yaw disagreement (deg)','inference'),
 'xy_overlap':('XY pseudo-distribution Bhattacharyya overlap','inference'),
 'top50_overlap':('Vision/acoustic top50 XY overlap','inference'),
 'v_scale_mean':('Vision mean predicted Laplace scale','inference'),
 'v_scale_relative':('Vision scale / predicted ray depth','inference'),
 'v_depth_mean':('Vision mean predicted ray depth (m)','inference'),
 'v_depth_cv':('Vision predicted depth coefficient of variation','inference'),
 'v_depth_range':('Vision predicted depth range (m)','inference'),
 'v_fit_at_map':('Vision normalized ray residual at its MAP','inference'),
 'v_map_wall_m':('Wall clearance at vision MAP (m)','inference'),
 'a_map_wall_m':('Wall clearance at acoustic MAP (m)','inference'),
 'v_map_corner_m':('Two orthogonal visible-wall proximity at vision MAP (m)','inference'),
 'a_map_corner_m':('Two orthogonal visible-wall proximity at acoustic MAP (m)','inference'),
 'audio_early_fraction':('Observed energy fraction before 50ms','inference'),
 'audio_low_fraction':('Observed energy fraction below 1.5kHz','inference'),
 'audio_mic_energy_cv':('Observed inter-microphone energy CV','inference'),
 'audio_coherence_mag':('Observed mean complex-coherency magnitude','inference'),
 'rgb_black_fraction':('RGB near-black pixel fraction (not no-hit)','inference'),
 'gt_wall_m':('GT wall clearance (m)','GT_diagnostic_only'),
 'gt_corner_m':('GT two orthogonal visible-wall proximity (m)','GT_diagnostic_only'),
 'gt_depth_mean':('Mean GT-nearest structural ray depth (m)','GT_diagnostic_only'),
 'v_gt_depth_mae':('Vision vs GT-nearest structural depth MAE (m)','GT_diagnostic_only'),
 'v_gt_depth_ratio':('Median predicted / GT structural depth','GT_diagnostic_only'),
 'furnished_gt_depth_mae':('Furnished vs GT-nearest structural depth MAE (m)','GT_diagnostic_only'),
 'v_furnished_depth_mae':('Vision vs furnished ray depth MAE (m)','GT_diagnostic_only'),
 'furnished_valid_fraction':('Fraction of finite furnished diagnostic rays','GT_diagnostic_only'),
 'furniture_occlusion_fraction':('Audited furnished occlusion fraction','GT_diagnostic_only'),
 'exterior_opening_fraction':('Audited exterior-opening first-hit fraction','GT_diagnostic_only'),
 'a_gt_local_rank':('Acoustic best <=1m region XY rank','GT_diagnostic_only'),
 'v_gt_local_rank':('Vision best <=1m region XY rank','GT_diagnostic_only'),
}


def spatial_stats(score,xy):
    """Fixed-temperature descriptive distributions, NOT calibrated confidence.

    XY uses max over yaw then zscore+softmax; normalized entropy comparable in
    support size, not proven calibrated across scenes. Repeat at T=0.5,2.
    """
    s=score.max(1).astype(float)
    z=standardized(s)
    top=np.argsort(-s,kind='stable')[:50]
    peak=int(top[0])
    dist=np.linalg.norm(xy-xy[peak],axis=1)
    separated=dist>1
    probability=softmax(z)
    entropy=lambda p:float(-(p*np.log(np.maximum(p,1e-300))).sum()/np.log(len(p)))
    modes=[]
    for i in top:
        if not modes or np.min(np.linalg.norm(xy[modes]-xy[i],axis=1))>1:
            modes.append(int(i))
    yaw_probability=softmax(standardized(score[peak]))
    return dict(entropy=entropy(probability),entropy_t05=entropy(softmax(z/.5)),
        entropy_t2=entropy(softmax(z/2)),peak_mass_1m=float(probability[~separated].sum()),
        separated_gap=float(z[peak]-z[separated].max()) if separated.any() else 0.,
        top50_modes=len(modes),top50_rms_m=float(np.sqrt(np.mean(np.sum((xy[top]-xy[top].mean(0))**2,axis=1)))),
        yaw_entropy=entropy(yaw_probability)),probability,top


def outcome_labels(record):
    ok={m:record['methods'][m]['xy_error_m']<=1 for m in METHODS}
    joint={m:ok[m] and record['methods'][m]['yaw_error_deg']<=30 for m in METHODS}
    v,a,f,k=[ok[m] for m in METHODS]
    return dict(**{m+'_ok':ok[m] for m in METHODS},**{m+'_joint':joint[m] for m in METHODS},
        vision_fail=not v,acoustic_fail=not a,both_fail=not v and not a,
        fusion_repair=not v and f,fusion_harm=v and not f,
        topk_repair=not v and k,topk_harm=v and not k,
        fusion_synergy=not v and not a and f,topk_synergy=not v and not a and k,
        fusion_only=f and not k,topk_only=k and not f,
        both_fusion_topk=f and k,neither_fusion_topk=not f and not k)


def geometry_features(root,scene):
    directory=Path(root)/'2DFP'/scene
    metadata=read(directory/'floorplan_metadata.json')
    free=np.asarray(Image.open(directory/'map.png').convert('L'))>127
    clearance=distance_transform_edt(free)*metadata['map_spacing_m']
    def at(points):
        b=metadata['world_bounds_m'];s=metadata['map_spacing_m']
        points=np.atleast_2d(points)
        col=np.rint((points[:,0]-b['x_min'])/s).astype(int)
        row=np.rint((points[:,1]-b['z_min'])/s).astype(int)
        if (row<0).any() or (row>=free.shape[0]).any() or (col<0).any() or (col>=free.shape[1]).any():
            raise ValueError('Query exceeds canonical map')
        return clearance[row,col]
    return at


def orthogonal_walls(field,cap=5.):
    """Identify TWO wall orientations, not merely two perpendicular rays.

    Three adjacent ray endpoints must form a locally straight wall patch. Pair
    patches with tangents 90+/-15 degrees apart. Return the smallest maximum
    perpendicular wall distance, capped at5m if no such nearby pair exists.
    Still a proxy: two visible walls need not meet at a physical room corner.
    """
    field=np.asarray(field,float)
    angle=np.arange(36)*np.pi/18
    point=field[...,None]*np.stack((np.cos(angle),np.sin(angle)),-1)
    previous,following=np.roll(point,1,axis=1),np.roll(point,-1,axis=1)
    first,second=point-previous,following-point
    length1=np.linalg.norm(first,axis=-1);length2=np.linalg.norm(second,axis=-1)
    tangent=following-previous
    tangent/=np.maximum(np.linalg.norm(tangent,axis=-1)[...,None],1e-12)
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    straight=(abs(cross(tangent,point-previous))<.05)
    straight&=((first*second).sum(-1)/np.maximum(length1*length2,1e-12)>np.cos(np.deg2rad(15)))
    straight&=(length1>.01)&(length2>.01)&(length1<2)&(length2<2)
    hit=(field>0)&(field<cap)
    straight&=hit&np.roll(hit,1,axis=1)&np.roll(hit,-1,axis=1)
    distance=abs(cross(tangent,point))
    result=np.full(len(field),cap)
    for i in range(36):
        dot=abs((tangent[:,i:i+1]*tangent).sum(-1))
        valid=straight[:,i:i+1]&straight&(dot<np.sin(np.deg2rad(15)))
        value=np.maximum(distance[:,i:i+1],distance)
        result=np.minimum(result,np.where(valid,value,cap).min(1))
    return result


def bootstrap_weights(records,draws=500):
    keys=[(r['scene_id'],*np.round(r['gt_pose_xzyaw'][:2],6)) for r in records]
    lut={key:i for i,key in enumerate(sorted(set(keys)))}
    clusters=np.array([lut[key] for key in keys])
    by_scene=defaultdict(list)
    for key,i in lut.items():by_scene[key[0]].append(i)
    rng=np.random.default_rng(42)
    weights=np.zeros((draws,len(lut)))
    for indices in by_scene.values():
        for b in range(draws):
            chosen=rng.choice(indices,len(indices),replace=True)
            weights[b]+=np.bincount(chosen,minlength=len(lut))
    return clusters,weights


def rate_interval(mask,success,clusters,weights):
    if not mask.any():return dict(n=0,positions=0,rate=None,ci95=None)
    count=np.bincount(clusters,weights=mask,minlength=weights.shape[1])
    hits=np.bincount(clusters,weights=mask&success,minlength=weights.shape[1])
    den=weights@count;num=weights@hits
    values=num[den>0]/den[den>0]
    return dict(n=int(mask.sum()),positions=int((count>0).sum()),rate=float(success[mask].mean()),
                ci95=np.quantile(values,[.025,.975]).tolist())


def paired_delta(mask,left,right,clusters,weights):
    if not mask.any():return dict(delta=None,ci95=None)
    count=np.bincount(clusters,weights=mask,minlength=weights.shape[1])
    delta=left.astype(float)-right.astype(float)
    total=np.bincount(clusters,weights=delta*mask,minlength=weights.shape[1])
    den=weights@count
    values=(weights@total)[den>0]/den[den>0]
    return dict(delta=float(delta[mask].mean()),ci95=np.quantile(values,[.025,.975]).tolist())


def auc(values,success):
    n=int(success.sum());m=len(success)-n
    if not n or not m:return None
    ranks=rankdata(values)
    return float((ranks[success].sum()-n*(n+1)/2)/(n*m))


def analyze(records):
    clusters,weights=bootstrap_weights(records)
    scenes=sorted({r['scene_id'] for r in records})
    features={key:np.array([r['features'][key] for r in records]) for key in FEATURES}
    labels={key:np.array([r['labels'][key] for r in records],bool) for key in records[0]['labels']}
    scene_ids=np.array([r['scene_id'] for r in records])
    conditions={'all':np.ones(len(records),bool),**labels}
    tables={}
    for name,mask in conditions.items():
        tables[name]=dict(n=int(mask.sum()),positions=len(set(clusters[mask])),
            methods={m:rate_interval(mask,labels[m+'_ok'],clusters,weights) for m in METHODS},
            medians={f:float(np.median(v[mask])) if mask.any() else None for f,v in features.items()})
    strata={};thresholds={}
    selected=['gt_wall_m','gt_corner_m','v_entropy','a_entropy','v_separated_gap','a_separated_gap','v_depth_mean','v_depth_range','v_scale_mean',
        'v_scale_relative','v_depth_cv','v_gt_depth_mae','furnished_gt_depth_mae','map_disagreement_m',
        'xy_overlap','audio_early_fraction','audio_coherence_mag','v_top50_rms_m','a_top50_rms_m',
        'v_map_wall_m','a_map_wall_m','furniture_occlusion_fraction','exterior_opening_fraction']
    for key in selected:
        values=features[key]
        if key=='gt_wall_m':
            bounds=[0,.5,1,2,float('inf')]
            masks=[(values>=lo)&(values<hi) for lo,hi in zip(bounds[:-1],bounds[1:])]
            names=['<0.5m','0.5–1m','1–2m','>=2m']
        elif key=='gt_corner_m':
            masks=[values<1,(values>=1)&(values<2),values>=2];names=['<1m','1–2m','>=2m']
        else:
            quantile=np.zeros(len(records),int)
            thresholds[key]={}
            for scene in scenes:
                ids=scene_ids==scene
                # Ties remain together; thresholds use inputs, never outcomes.
                edges=np.quantile(values[ids],[.25,.5,.75])
                thresholds[key][scene]=edges.tolist()
                quantile[ids]=np.searchsorted(edges,values[ids],side='left')
            masks=[quantile==i for i in range(4)];names=['Q1 low','Q2','Q3','Q4 high']
        strata[key]=[]
        for name,mask in zip(names,masks):
            bin_index=len(strata[key])
            for index in np.flatnonzero(mask):
                records[int(index)].setdefault('strata',{})[key]=bin_index
            strata[key].append(dict(bin=name,n=int(mask.sum()),positions=len(set(clusters[mask])),
                methods={m:rate_interval(mask,labels[m+'_ok'],clusters,weights) for m in METHODS},
                fusion_repair=int((mask&labels['fusion_repair']).sum()),fusion_harm=int((mask&labels['fusion_harm']).sum()),
                paired_delta_vs_vision={m:paired_delta(mask,labels[m+'_ok'],labels['vision_only_ok'],clusters,weights) for m in METHODS[1:]},
                fusion_repair_rate_given_vfail=float(labels['fusion_repair'][mask&labels['vision_fail']].mean()) if (mask&labels['vision_fail']).any() else None,
                fusion_harm_rate_given_vsuccess=float(labels['fusion_harm'][mask&~labels['vision_fail']].mean()) if (mask&~labels['vision_fail']).any() else None,
                per_scene={s:{m:float(labels[m+'_ok'][mask&(scene_ids==s)].mean()) if (mask&(scene_ids==s)).any() else None for m in METHODS} for s in scenes}))
    ambiguity=[]
    high={}
    for modality in ('v','a'):
        high[modality]=np.zeros(len(records),bool)
        for scene in scenes:
            ids=scene_ids==scene
            high[modality][ids]=features[modality+'_entropy'][ids]>=np.median(features[modality+'_entropy'][ids])
    for v in (False,True):
        for a in (False,True):
            mask=(high['v']==v)&(high['a']==a)
            ambiguity.append(dict(vision_ambiguity='high' if v else 'low',acoustic_ambiguity='high' if a else 'low',n=int(mask.sum()),
                methods={m:rate_interval(mask,labels[m+'_ok'],clusters,weights) for m in METHODS},
                fusion_repair=int((mask&labels['fusion_repair']).sum()),topk_repair=int((mask&labels['topk_repair']).sum()),
                fusion_synergy=int((mask&labels['fusion_synergy']).sum()),topk_synergy=int((mask&labels['topk_synergy']).sum())))
    # Single-feature associations, not a fitted classifier or selected gate.
    associations=[]
    tasks=[('vision_success',np.ones(len(records),bool),labels['vision_only_ok']),
        ('acoustic_success',np.ones(len(records),bool),labels['acoustic_only_ok']),
        ('fusion_repair_given_vfail',labels['vision_fail'],labels['equal_weight_fusion_ok']),
        ('fusion_harm_given_vsuccess',~labels['vision_fail'],~labels['equal_weight_fusion_ok']),
        ('topk_repair_given_vfail',labels['vision_fail'],labels['visual_topk_acoustic_rerank_ok'])]
    for task,mask,y in tasks:
        for key,x in features.items():
            per_scene={s:auc(x[mask&(scene_ids==s)],y[mask&(scene_ids==s)]) for s in scenes}
            valid=[v for v in per_scene.values() if v is not None]
            associations.append(dict(task=task,feature=key,availability=FEATURES[key][1],
                auc_higher_feature_predicts_event=auc(x[mask],y[mask]),per_scene=per_scene,
                mean_scene_auc=float(np.mean(valid)) if valid else None,
                sign_consistency=sum(v>.5 for v in valid),scene_count=len(valid)))
    sensitivity=[]
    for temperature,key in [(.5,'entropy_t05'),(1.,'entropy'),(2.,'entropy_t2')]:
        both_high=np.ones(len(records),bool)
        for modality in ('v','a'):
            value=np.array([r['features'][modality+'_'+key] for r in records])
            above=np.zeros(len(records),bool)
            for scene in scenes:
                ids=scene_ids==scene
                above[ids]=value[ids]>=np.median(value[ids])
            both_high&=above
        sensitivity.append(dict(temperature=temperature,n=int(both_high.sum()),
            methods={m:rate_interval(both_high,labels[m+'_ok'],clusters,weights) for m in METHODS}))
    transitions=[]
    for v in (False,True):
        for a in (False,True):
            mask=(~labels['vision_fail']==v)&(~labels['acoustic_fail']==a)
            transitions.append(dict(vision_ok=v,acoustic_ok=a,n=int(mask.sum()),
                fusion_success=int((mask&labels['equal_weight_fusion_ok']).sum()),
                topk_success=int((mask&labels['visual_topk_acoustic_rerank_ok']).sum())))
    upper=dict(vision_or_acoustic=float((labels['vision_only_ok']|labels['acoustic_only_ok']).mean()),
        vision_or_equal_fusion=float((labels['vision_only_ok']|labels['equal_weight_fusion_ok']).mean()),
        vision_or_topk=float((labels['vision_only_ok']|labels['visual_topk_acoustic_rerank_ok']).mean()),
        any_four=float(np.logical_or.reduce([labels[m+'_ok'] for m in METHODS]).mean()),
        caveat='GT-dependent upper bounds, not implementable policies or measured learned-gate results')
    return dict(queries=len(records),scenes=scenes,physical_positions=weights.shape[1],
        groups=tables,strata=strata,bin_thresholds=thresholds,ambiguity=ambiguity,associations=associations,transitions=transitions,
        oracle_upper_bounds=upper,per_scene={s:{m:float(labels[m+'_ok'][scene_ids==s].mean()) for m in METHODS} for s in scenes},
        joint={m:float(labels[m+'_joint'].mean()) for m in METHODS},
        ambiguity_temperature_sensitivity=sensitivity,
        topk_exclusion=sum(not r['diagnostics']['topk_oracle_1m'] for r in records),
        topk_visual_yaw_oracle=sum(r['diagnostics']['topk_with_visual_yaw_oracle_1m_30deg'] for r in records),
        feature_dictionary={k:dict(label=v[0],availability=v[1]) for k,v in FEATURES.items()},
        protocol=dict(test_fitting=False,new_training=False,new_rir=False,criterion='XY<=1m; joint additionally yaw<=30deg',
            entropy='softmax(zscore(max_yaw score)/T), T=1; descriptive not calibrated; sensitivity T=.5,2',
            bins='within-scene quartiles except explicit metric wall/corner bins; ties stay together',
            confidence_intervals='500 paired physical-pose cluster bootstrap draws, stratified by these 3 scenes; not unseen-scene CI',
            associations='AUC with raw feature direction, not tuned/flipped; no significance claims or multiple-testing correction',
            geometry='Wall clearance uses canonical map. Orthogonal-wall proxy fits locally straight patches to 3 neighboring ray endpoints, pairs patch tangents 90+/-15deg, and minimizes max perpendicular distance; capped5m. Two walls need not meet at a physical corner.',
            caveats=['Exploratory reused 3 test scenes; cannot train or choose deployment gate on these labels',
                'Vision trained116 scenes, acoustic12; scene confounding examined per scene',
                'Clutter/geometry flags are not confirmed rendering errors; no data excluded',
                'Zero or swapped audio is not a map-only trained baseline',
                'RGB darkness is not no-hit or exterior evidence']))


def mechanism_checks(records,old_report):
    banks=read(Path(old_report)/'banks.json')
    for r in records:
        cells=banks[r['scene_id']]['cells']
        top=np.fromfile(Path(old_report)/r['heatmap'],dtype=np.uint8).reshape(4,cells)[3]>0
        if int(top.sum())!=50:raise ValueError('Frozen top50 heat mask changed')
        r['diagnostics']['acoustic_map_in_visual_top50']=bool(top[r['methods']['acoustic_only']['cell']])
        r['diagnostics']['fusion_map_in_visual_top50']=bool(top[r['methods']['equal_weight_fusion']['cell']])
    subset=lambda name:[r for r in records if r['labels'][name]]
    topk=subset('topk_synergy');fusion=subset('fusion_synergy')
    fusion_only=subset('fusion_only');topk_only=subset('topk_only')
    both_good=[r for r in records if r['labels']['vision_only_ok'] and r['labels']['acoustic_only_ok']]
    both_good_harmed=[r for r in both_good if not r['labels']['equal_weight_fusion_ok']]
    both_good_preserved=[r for r in both_good if r['labels']['equal_weight_fusion_ok']]
    return dict(topk_synergy=dict(n=len(topk),acoustic_global_map_removed=sum(not r['diagnostics']['acoustic_map_in_visual_top50'] for r in topk),
        median_global_audio_gt_region_rank=float(np.median([r['features']['a_gt_local_rank'] for r in topk])),
        median_global_vision_gt_region_rank=float(np.median([r['features']['v_gt_local_rank'] for r in topk]))),
        fusion_synergy=dict(n=len(fusion),median_global_audio_gt_region_rank=float(np.median([r['features']['a_gt_local_rank'] for r in fusion])),
        median_global_vision_gt_region_rank=float(np.median([r['features']['v_gt_local_rank'] for r in fusion]))),
        fusion_only=dict(n=len(fusion_only),gt_neighborhood_excluded_by_top50=sum(not r['diagnostics']['topk_oracle_1m'] for r in fusion_only),
            selected_cell_outside_visual_top50=sum(not r['diagnostics']['fusion_map_in_visual_top50'] for r in fusion_only)),
        topk_only=dict(n=len(topk_only),acoustic_global_map_removed=sum(not r['diagnostics']['acoustic_map_in_visual_top50'] for r in topk_only)),
        both_unimodal_xy_good=dict(n=len(both_good),fusion_harmed=len(both_good_harmed),
            median_yaw_disagreement_harmed=float(np.median([r['features']['map_yaw_disagreement_deg'] for r in both_good_harmed])),
            median_yaw_disagreement_preserved=float(np.median([r['features']['map_yaw_disagreement_deg'] for r in both_good_preserved])),
            caveat='Small harmed cohort; association with yaw mismatch, not a causal intervention'),
        yaw=dict(**{m:dict(xy_success=sum(r['labels'][m+'_ok'] for r in records),
            yaw_fail_given_xy_success=sum(r['labels'][m+'_ok'] and not r['labels'][m+'_joint'] for r in records)) for m in METHODS}),
        note='Decision-path checks, not proof of physical clutter causation or a learned reliability gate')


def refresh(args):
    report=Path(args.report)
    records=read(report/'records.json');prior=read(report/'analysis.json')
    if sha(Path(args.old_report)/'records.json')!=prior['provenance']['original_records_sha256']:
        raise ValueError('Original report changed')
    banks=read(Path(args.old_report)/'banks.json')
    for scene,bank in banks.items():
        payload=np.load(Path(args.vision_root)/'UnLoc/structural_targets'/scene/'desdf.npy',allow_pickle=True).item()
        xy=np.asarray(bank['xy']);field=bank_windows(payload,xy)[:,np.arange(36),5]
        proxy=orthogonal_walls(field)
        for r in records:
            if r['scene_id']!=scene:continue
            closest=int(np.linalg.norm(xy-np.asarray(r['gt_pose_xzyaw'][:2]),axis=1).argmin())
            r['features']['gt_corner_m']=float(proxy[closest])
            for key,method in [('v_map_corner_m','vision_only'),('a_map_corner_m','acoustic_only')]:
                r['features'][key]=float(proxy[r['methods'][method]['cell']])
    mechanisms=mechanism_checks(records,args.old_report)
    result=analyze(records);result['mechanisms']=mechanisms
    result['provenance']=prior['provenance']
    result['provenance']['analysis_source_sha256']=sha(__file__)
    atomic_json(report/'analysis.json',result);atomic_json(report/'records.json',records)
    fields=['scene_id','sample_index',*FEATURES,*records[0]['labels']]
    temporary_csv=report/'features.csv.new'
    with temporary_csv.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:writer.writerow(dict(scene_id=r['scene_id'],sample_index=r['sample_index'],
            **{k:r['features'][k] for k in FEATURES},**r['labels']))
    temporary_csv.replace(report/'features.csv')
    temporary=report/'index.html.new'
    shutil.copyfile(Path(__file__).with_name('templates')/'reliability_analysis.html',temporary)
    temporary.replace(report/'index.html')


def run(args):
    root,evaluation,report=Path(args.root),Path(args.evaluation),Path(args.report)
    report.mkdir(parents=True,exist_ok=True)
    old=Path(args.old_report);original=read(old/'records.json');summary=read(old/'summary.json')
    lookup={(r['scene_id'],r['sample_index']):r for r in original}
    index=read(evaluation/'bank_index.json');ready=read(evaluation/'ready.json')
    if sha(evaluation/'bank_index.json')!=summary['bank_sha256']:
        raise ValueError('Frozen candidate bank mismatch')
    checkpoint=summary['acoustic_checkpoint_sha256']
    records=[];provenance={};started=time.time()
    for entry in index['scenes']:
        scene=entry['scene_id'];xy=np.asarray(entry['candidate_xz'])
        candidate_path=evaluation/f'{scene}_candidate_embeddings_{checkpoint[:12]}.npy'
        observed_path=evaluation/f'{scene}_observed_embeddings_{checkpoint[:12]}.npy'
        candidate=np.load(candidate_path,mmap_mode='r');observed=np.load(observed_path,mmap_mode='r')
        queries=read(root/entry['observation_schedule_path'])['queries']
        audio_scores=observed@candidate.reshape(-1,candidate.shape[-1]).T
        if audio_scores.shape!=(len(queries),len(xy)*36):raise ValueError('Embedding schedule shape changed')
        target_path=Path(args.vision_root)/'UnLoc/structural_targets'/scene/'desdf.npy'
        payload=np.load(target_path,allow_pickle=True).item();windows=bank_windows(payload,xy)
        field=windows[:,np.arange(36),5]
        corner=orthogonal_walls(field)
        wall_at=geometry_features(root,scene);wall=wall_at(xy)
        rowpath=Path(args.vision_eval)/'rows'/f'{scene}.json';depthpath=Path(args.vision_eval)/'depth'/f'{scene}.npz'
        rows=read(rowpath);rowlut={r['sample_index']:i for i,r in enumerate(rows)}
        with np.load(depthpath) as data:depth=data['location'].copy();scale=data['scale'].copy()
        # frame_adapter/.../depth40.npy is STRUCTURAL training supervision, not
        # furnished sensor depth. Use the original furnished audit ledger and
        # accepted replacement ledger, matched by pose and sample ID.
        audit_root=Path(args.vision_root)/'UnLoc/furnished_depth_qa_v2_nohit_only'
        audit_path=audit_root/'scenes'/f'{scene}.jsonl'
        replacement_path=audit_root/'replacements'/scene/'replacement_manifest.json'
        furnished={r['sample_index']:r for r in [json.loads(line) for line in audit_path.read_text().splitlines()]}
        for replacement in read(replacement_path)['records']:
            if replacement.get('accepted'):
                furnished[replacement['sample_index']]=replacement
        receipt_path=root/ready['features'][scene]['Observed']['path']
        if sha(receipt_path)!=ready['features'][scene]['Observed']['sha256']:raise ValueError('Observed feature receipt changed')
        receipt=read(receipt_path)
        for name,digest in receipt['files'].items():
            if sha(receipt_path.parent/name)!=digest:raise ValueError('Observed feature cache changed')
        energy=np.load(receipt_path.parent/'energy.npy',mmap_mode='r')
        coherence=np.load(receipt_path.parent/'coherence.npy',mmap_mode='r')
        audio_lut={p:i for i,p in enumerate(receipt['paths'])}
        for i,q in enumerate(queries):
            source=lookup[(scene,q['sample_index'])];j=rowlut[q['sample_index']]
            rays,scales=angular_rays(depth[j],scale[j])
            v=-(abs(windows-rays)/scales).sum(-1)/40
            a=audio_scores[i].reshape(len(xy),36)
            chosen,top=select(v,a,50)
            for method,(cell,yaw) in chosen.items():
                prior=source['methods'][method]
                if (cell,yaw)!=(prior['cell'],prior['yaw_bin']):
                    raise ValueError(f'Frozen decision did not reproduce: {scene}/{q["sample_index"]}/{method}')
            vs,pv,vt=spatial_stats(v,xy);ast,pa,at=spatial_stats(a,xy)
            f={**{'v_'+k:val for k,val in vs.items()},**{'a_'+k:val for k,val in ast.items()}}
            vc,vy=chosen['vision_only'];ac,ay=chosen['acoustic_only']
            gt=np.asarray(source['gt_pose_xzyaw']);dist=np.linalg.norm(xy-gt[:2],axis=1);near=dist<=1
            nearest=int(dist.argmin());heading=int(round(gt[2]/10))%36
            gt_rays=windows[nearest,heading]
            if not np.allclose(gt_rays,source['gt_nearest_fp_rays']):raise ValueError('GT-nearest ray order changed')
            f.update(map_disagreement_m=float(np.linalg.norm(xy[vc]-xy[ac])),map_yaw_disagreement_deg=abs((vy*10-ay*10+180)%360-180),
                xy_overlap=float(np.sqrt(pv*pa).sum()),top50_overlap=len(set(vt)&set(at))/50,
                v_scale_mean=float(scales.mean()),v_scale_relative=float(np.mean(scales/np.maximum(rays,.05))),
                v_depth_mean=float(rays.mean()),v_depth_cv=float(rays.std()/max(rays.mean(),.05)),v_depth_range=float(np.ptp(rays)),
                v_fit_at_map=float(np.mean(abs(windows[vc,vy]-rays)/scales)),
                v_map_wall_m=float(wall[vc]),a_map_wall_m=float(wall[ac]),
                v_map_corner_m=float(corner[vc]),a_map_corner_m=float(corner[ac]),
                gt_wall_m=float(wall_at(gt[:2])[0]),gt_corner_m=float(corner[nearest]),gt_depth_mean=float(gt_rays.mean()),
                v_gt_depth_mae=float(np.mean(abs(rays-gt_rays))),v_gt_depth_ratio=float(np.median(rays/np.maximum(gt_rays,.05))))
            # Furnished sensor returns camera-forward depth. Convert to the same
            # angular slant-range representation as structural/visual rays.
            audit=furnished[q['sample_index']]
            if not np.allclose(audit['pose_xyz_m'],q['centre_xyz_m'],rtol=0,atol=1e-5):
                raise ValueError('Furnished audit pose mismatch')
            furnished40=np.array([np.nan if x is None else x for x in audit['furnished_depth40_m']])
            furnished_rays,_=angular_rays(furnished40,np.ones_like(furnished40))
            valid_furnished=np.isfinite(furnished_rays)&(furnished_rays>0)
            if not valid_furnished.any():raise ValueError('No usable furnished diagnostic rays')
            f.update(furnished_gt_depth_mae=float(np.mean(abs(furnished_rays[valid_furnished]-gt_rays[valid_furnished]))),
                v_furnished_depth_mae=float(np.mean(abs(rays[valid_furnished]-furnished_rays[valid_furnished]))),
                furnished_valid_fraction=float(valid_furnished.mean()),
                furniture_occlusion_fraction=audit.get('counts',{}).get('furniture_occlusion',0)/40,
                exterior_opening_fraction=audit.get('counts',{}).get('exterior_opening_first_hits',0)/40)
            for prefix,score in [('v',v),('a',a)]:
                local=float(score.max(1)[near].max())
                f[prefix+'_gt_local_rank']=1+int((score.max(1)>local).sum())
            rid=audio_lut[q['observed_rir_path']]
            power=np.expm1(np.array(energy[rid],dtype=float))/100
            times=2+np.arange(power.shape[-1])*64/48000*1000
            frequencies=np.arange(3,3+power.shape[1])*48000/256
            c=np.array(coherence[rid],float)
            mic_mass=power.sum((1,2))
            f.update(audio_early_fraction=float(power[...,times<50].sum()/power.sum()),
                audio_low_fraction=float(power[:,frequencies<1500].sum()/power.sum()),
                audio_mic_energy_cv=float(mic_mass.std()/mic_mass.mean()),
                audio_coherence_mag=float(np.sqrt(np.sum(c*c,axis=2)).mean()))
            image=np.asarray(Image.open(Path(args.vision_root)/q['image_path']).convert('RGB').resize((160,120)))
            f['rgb_black_fraction']=float((image.max(-1)<8).mean())
            if not np.isfinite(list(f.values())).all():raise ValueError('Nonfinite diagnostic feature')
            result=dict(source,features=f,labels=outcome_labels(source),
                furnished_rays=[float(x) if np.isfinite(x) else None for x in furnished_rays])
            records.append(result)
        provenance[scene]=dict(candidate_embeddings_sha256=sha(candidate_path),observed_embeddings_sha256=sha(observed_path),
            rows_sha256=sha(rowpath),depth_sha256=sha(depthpath),desdf_sha256=sha(target_path),
            furnished_audit_sha256=sha(audit_path),replacement_ledger_sha256=sha(replacement_path),
            observation_schedule_sha256=sha(root/entry['observation_schedule_path']),queries=len(queries),all_four_decisions_reproduced=True)
        atomic_json(report/'progress.json',dict(completed_scenes=len(provenance),queries=len(records),elapsed_seconds=time.time()-started))
        print(scene,len(queries),'four-way exact reproduction OK',flush=True)
    if len(records)!=len(original):raise ValueError('Cohort count changed')
    records.sort(key=lambda r:(r['scene_id'],r['sample_index']))
    result=analyze(records)
    result['provenance']=dict(original_summary_sha256=sha(old/'summary.json'),original_records_sha256=sha(old/'records.json'),
        acoustic_checkpoint_sha256=checkpoint,acoustic_epoch=summary['acoustic_epoch'],scenes=provenance)
    atomic_json(report/'analysis.json',result);atomic_json(report/'records.json',records)
    fields=['scene_id','sample_index',*FEATURES,*records[0]['labels']]
    with (report/'features.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:writer.writerow(dict(scene_id=r['scene_id'],sample_index=r['sample_index'],
            **{k:r['features'][k] for k in FEATURES},**r['labels']))
    temporary=report/'index.html.new'
    shutil.copyfile(Path(__file__).with_name('templates')/'reliability_analysis.html',temporary)
    temporary.replace(report/'index.html')
    atomic_json(report/'status.json',dict(status='complete',queries=len(records),test_fitting=False,new_training=False,
        all_four_decisions_reproduced=True,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','evaluation','old-report','vision-root','vision-eval','report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--aggregate-only',action='store_true')
    args=p.parse_args()
    if not args.aggregate_only:run(args)
    refresh(args)
