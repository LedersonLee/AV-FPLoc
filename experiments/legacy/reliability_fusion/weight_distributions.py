"""Read-only source export: learned Val weights and GT-optimized Train weights.

Creates derived local CSV/plots only. Never opens Test data or changes reports,
checkpoints, source datasets, or any learned weights.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from acoustic_branch.contracts import atomic_json, sha
from .input_supervision_audit import read


def describe(weights):
    a = np.asarray(weights)
    return dict(queries=len(a), acoustic_mean=float(a.mean()), vision_mean=float(1-a.mean()),
        std=float(a.std()), **{k:float(v) for k,v in zip(('min','p10','p25','median','p75','p90','max'), np.quantile(a,[0,.1,.25,.5,.75,.9,1]))},
        exact_zero=int((a==0).sum()), exact_one=int((a==1).sum()))


def export(path, records, scope):
    fields=['scope','scene_id','sample_index','method','acoustic_weight','vision_weight','xy_error_m','yaw_error_deg']
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in records:
            g=np.asarray(r.get('gt', r.get('truth')))
            for name,m in r['methods'].items():
                pp=np.asarray(m['pose']); w=m['weight']
                writer.writerow(dict(scope=scope,scene_id=r['scene_id'],sample_index=r['sample_index'],method=name,
                    acoustic_weight=w,vision_weight=1-w,xy_error_m=float(np.linalg.norm(pp[:2]-g[:2])),
                    yaw_error_deg=float(abs((pp[2]-g[2]+180)%360-180))))


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    vp=args.ssd/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/fusion-subset12-logit/records.json'
    tp=args.root/'ReliabilityFusion/pilots/independent_pose_risk_v1/verified_records.json'
    val,train=read(vp),read(tp)
    assert len(val)==1704 and len({r['scene_id'] for r in val})==3
    assert len(train)==6387 and {r['split'] for r in train}=={'train'}
    last=['logit_regression_seed42_last','logit_regression_seed43_last','success_set_seed42_last','success_set_seed43_last']
    arrays=[np.array([r['methods'][n]['weight'] for r in val]) for n in last]
    best=[n for n in val[0]['methods'] if n.endswith('_best')]
    for n in best:
        np.testing.assert_allclose([r['methods'][n]['weight'] for r in val], .1, atol=1e-7,rtol=0)
    summary=dict(learned_val={n:describe(a) for n,a in zip(last,arrays)},
        independent_gt_train={n:describe([r['methods'][n]['weight'] for r in train]) for n in train[0]['methods']},
        best_notice='All Best checkpoints are epoch0, constant acousticw.1. Histograms below are Last40 only.',
        oracle_notice='Success oracle keepsw.1 on baseline-correct and joint-unrecoverable queries; its concentration at.1 is by construction, not learned evidence.',
        coefficient_notice='w is acoustic score coefficient; vision=1-w. Not calibrated modality correctness probability.',
        input_sha256={str(vp):sha(vp),str(tp):sha(tp)},test_accessed=False)
    atomic_json(out/'summary.json',summary)
    export(out/'learned_val_query_weights.csv',val,'learned_Val3')
    export(out/'independent_gt_train_query_weights.csv',train,'GT_optimized_Train12_NOT_learned')
    labels=['Logit s42','Logit s43','Success-set s42','Success-set s43']
    colors=['#245bad','#6699d4','#b25039','#de9678']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={'width_ratios':[1.1,1]},constrained_layout=True)
    for a,label,color in zip(arrays,labels,colors):
        axes[0].hist(a,bins=np.linspace(0,1,41),weights=np.ones(len(a))*100/len(a),histtype='step',linewidth=1.8,label=label,color=color)
    axes[0].set(xlim=(0,1),xlabel='Acoustic coefficient w (vision = 1 - w)',ylabel='Queries per 0.025-wide bin (%)',title='Learned Last40 weights | Val3, n=1,704')
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    b=axes[1].boxplot(arrays,vert=False,labels=labels,whis=(10,90),showfliers=False,patch_artist=True)
    for patch,color in zip(b['boxes'],colors):
        patch.set_facecolor(color);patch.set_alpha(.4)
    axes[1].scatter([a.mean() for a in arrays],np.arange(1,5),marker='D',s=28,c='#202c40',label='Mean',zorder=5)
    axes[1].set(xlim=(0,.8),xlabel='Acoustic coefficient w',title='Box: P25-P75; whiskers: P10-P90')
    axes[1].invert_yaxis();axes[1].legend(fontsize=9,loc='lower right');axes[1].grid(axis='x',alpha=.2)
    for ax in axes:
        ax.axvline(.1,ls='--',color='#555',lw=1)
    fig.savefig(out/'learned_val_weight_distributions.png',dpi=170);plt.close(fig)
    print(out/'learned_val_weight_distributions.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
