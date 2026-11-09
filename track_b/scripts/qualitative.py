"""Fixed first validation query per scene; no success-based case selection."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from track_b.scripts.pilot import materialize
from track_b.models.dense import DenseScorer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(2)
    protocol=json.loads((args.pilot/'protocol.json').read_text());fig,axes=plt.subplots(3,2,figsize=(10,12),layout='constrained');records=[]
    for row,sid in enumerate(protocol['val']):
        scene=materialize(args.root,sid,'val',16);rgb,rir,gt,qid=scene['data'][0];p=scene['p'];xy=p[::12,:2].cpu().numpy()
        meta=json.loads((args.root/f'2DFP/{sid}/floorplan_metadata.json').read_text());bounds=meta['world_bounds_m'];fp=np.array(Image.open(args.root/f'2DFP/{sid}/map.png'))
        for col,mode in enumerate(('visual','joint')):
            model=DenseScorer(mode=mode).cuda().eval();model.load_state_dict(torch.load(args.pilot/f'{mode}_42.pt',weights_only=True))
            with torch.no_grad():s=model(rgb,rir,scene['fp'],scene['cal'],p)[0];marg=s.reshape(-1,12).logsumexp(1).cpu().numpy();idx=s.argmax();top=s.topk(20).indices.cpu().numpy()
            err=float((p[idx,:2]-gt[0,:2]).norm());yaw=float(torch.atan2(torch.sin(p[idx,2]-gt[0,2]),torch.cos(p[idx,2]-gt[0,2])).abs()*180/np.pi)
            ax=axes[row,col];ax.imshow(fp,cmap='gray',vmin=0,vmax=255,extent=[bounds['x_min'],bounds['x_max'],bounds['z_max'],bounds['z_min']],alpha=.7)
            im=ax.scatter(xy[:,0],xy[:,1],c=marg-marg.max(),cmap='magma',s=8,vmin=-8,vmax=0)
            poses=p.cpu().numpy();truth=gt[0].cpu().numpy();ax.scatter(poses[top,0],poses[top,1],s=35,facecolors='none',edgecolors='cyan',linewidths=.6)
            ax.scatter([truth[0]],[truth[1]],c='lime',marker='*',s=140,edgecolors='black',label='GT');pred=poses[int(idx)]
            ax.arrow(pred[0],pred[1],np.cos(pred[2]),np.sin(pred[2]),color='cyan',width=.07,length_includes_head=True)
            ax.set_title(f'{mode} | {sid} / query {qid}\nMAP error {err:.2f}m / {yaw:.0f}deg');ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Z (m)')
            records.append(dict(scene=sid,query=qid,mode=mode,seed=42,translation_m=err,yaw_deg=yaw))
    fig.colorbar(im,ax=axes.ravel().tolist(),label='xy log-sum-exp score relative to maximum');fig.suptitle('Fixed development examples, no case cherry-picking\nGreen star: GT; cyan: raw Top20 and MAP heading; no discriminator')
    fig.savefig(args.out,dpi=130);plt.close(fig);args.out.with_suffix('.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
