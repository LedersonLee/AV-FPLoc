"""Predefined and worst-MAE inner-validation examples, including categorical heatmaps."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .prepare_targets import read


def main():
    p=argparse.ArgumentParser()
    for k in ['root','targets','run','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    protocol=read(a.run/'protocol.json');scenes=protocol['scene_ids']['val']
    selections=[('predefined',scenes[0],0)]
    errors=[]
    for scene in scenes:
        z=np.load(a.run/'spatial_categorical_42'/f'{scene}__normal.npz')
        err=(abs(z['mean']-z['target'])*z['mask']).sum(-1)/z['mask'].sum(-1).clip(1)
        errors.extend((float(e),scene,i) for i,e in enumerate(err))
    _,scene,i=max(errors);selections.append(('worst_spatial_categorical_seed42',scene,i))
    selection_rows=[]
    for label,scene,i in selections:
        refs=read(a.targets/f'{scene}_queries.json');q=refs[i]
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        axes[0].imshow(Image.open(a.root/q['rgb_path']));axes[0].axis('off');axes[0].set_title('Furnished RGB (reference only)')
        for arch,head in [('echo','regression'),('echo','categorical'),('spatial','regression'),('spatial','categorical')]:
            z=np.load(a.run/f'{arch}_{head}_42'/f'{scene}__normal.npz')
            axes[1].plot(np.arange(36)*10,z['mean'][i],label=arch+'/'+head,alpha=.8)
        truth=np.where(z['mask'][i],z['target'][i],np.nan)
        axes[1].plot(np.arange(36)*10,truth,'k.-',label='Structural GT (valid only)',linewidth=2)
        axes[1].set(xlabel='Body bearing (degrees)',ylabel='Range (m)',ylim=(0,20));axes[1].legend(fontsize=7)
        im=axes[2].imshow(z['probability'][i].T,origin='lower',aspect='auto',extent=[0,360,0,20],cmap='magma')
        axes[2].plot(np.arange(36)*10+5,truth,'c.',markersize=4)
        axes[2].set(xlabel='Body bearing (degrees)',ylabel='Range (m)',title='Spatial categorical probabilities + GT')
        fig.colorbar(im,ax=axes[2],label='Probability per0.25m bin')
        fig.suptitle(label+' / '+scene+' / query '+str(q['query_id'])+' / seed42 / inner validation')
        fig.tight_layout();fig.savefig(a.out/(label+'.png'),dpi=130);plt.close(fig)
        selection_rows.append(dict(selection=label,scene=scene,query_id=q['query_id'],query_index=i))
    (a.out/'selection.json').write_text(json.dumps(selection_rows,indent=2))


if __name__=='__main__':main()
