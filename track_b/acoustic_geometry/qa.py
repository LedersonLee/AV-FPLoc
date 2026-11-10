"""Predefined first three training scenes: stage rays, native RGB and map overlay."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    for k in ('root','targets','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    protocol=json.loads((a.targets/'protocol.json').read_text())
    for scene in protocol['scenes']['train'][:3]:
        z=np.load(a.targets/f'{scene}.npz');q=json.loads((a.targets/f'{scene}_queries.json').read_text())[0]
        meta=json.loads((a.root/f'2DFP/{scene}/floorplan_metadata.json').read_text())
        fp=np.array(Image.open(a.root/f'2DFP/{scene}/map.png'));b=meta['world_bounds_m'];spacing=meta['map_spacing_m']
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        axs[0].imshow(Image.open(a.root/q['rgb_path']));axs[0].set_title('Native furnished RGB');axs[0].axis('off')
        extent=[b['x_min'],b['x_min']+fp.shape[1]*spacing,b['z_min']+fp.shape[0]*spacing,b['z_min']]
        axs[1].imshow(fp,cmap='gray',extent=extent)
        seg=z['segments_'+str(z['heights'][0])]
        from matplotlib.collections import LineCollection
        axs[1].add_collection(LineCollection(seg,colors='tab:blue',linewidths=.4))
        pose=z['poses'][0];angle=pose[2]+np.arange(36)*2*np.pi/36
        end=pose[:2]+np.stack([np.cos(angle),np.sin(angle)],-1)*z['depth'][0,:,None]
        rays=np.stack([np.broadcast_to(pose[:2],end.shape),end],1)
        axs[1].add_collection(LineCollection(rays[z['mask'][0]],colors='tab:red',linewidths=.8))
        axs[1].plot(*pose[:2],'go');axs[1].set_title('Stage sections (blue), valid rays (red)')
        axs[2].plot(np.arange(36)*10,np.where(z['mask'][0],z['depth'][0],np.nan),'o-')
        axs[2].set(xlabel='Body bearing (degrees)',ylabel='Structural stage range (m)',ylim=(0,20),title='Masked missing hits')
        fig.suptitle(scene+' / query '+str(q['query_id']));fig.tight_layout();fig.savefig(a.out/f'{scene}.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()
