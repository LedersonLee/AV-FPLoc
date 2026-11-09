"""Plot completed measured aggregates only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    runs=['pilot_v1','pilot_long_v1','pilot_rays_v1','pilot_zero_v1'];colors=['#557799','#bb7733','#228866','#aa5588']
    for run,color in zip(runs,colors):
        p=a.runs/run/'summary.json'
        if not p.exists():continue
        rows=json.loads(p.read_text());modes=list(dict.fromkeys(r['mode'] for r in rows))
        labels=[f"{run.replace('pilot_','').replace('_v1','')}:{m}" for m in modes]
        for ax,metric in zip(axes,['joint','raw20']):
            for label,m in zip(labels,modes):
                vals=[r['normal'][metric]*100 for r in rows if r['mode']==m]
                ax.scatter(vals,[label]*len(vals),s=28,color=color,alpha=.65)
                ax.scatter([np.mean(vals)],[label],marker='|',s=180,color='black')
    for ax,title in zip(axes,['MAP joint success (%)','Raw Top20 joint coverage (%)']):
        ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_xlim(left=-.5)
    fig.suptitle('Exploratory: 3 development scenes / 48 queries; dots = seeds, black = mean\n0.5m / 12 yaw bins; not independent test; no FP-RIR reference')
    fig.savefig(a.out,dpi=170);plt.close(fig)

if __name__=='__main__':main()
