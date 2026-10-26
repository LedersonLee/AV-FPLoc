"""Plot controlled reflection-energy traces without synthesizing audio."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import read


def run(root,directory,source,output,rgb_root):
    output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('grid_plot',source)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    c=m.GridScoreConfig(sample_rate_hz=48000,direct_guard_samples=96,usable_samples=6144,nfft=256,hop=64,
        bands=((0,500),(500,1500),(1500,4000)),normalise='sum',distance='symmetric',shape_feature=True,feature='stft_band')
    for scene,queries in read(directory/'plan.json')['scenes'].items():
        for q in queries:
            sample=q['sample_index'];key=f'{scene}_{sample:05d}'
            refs=dict(furnished=root/q['observed']['path'],FP=root/q['candidates'][0]['path'],
                      native_stage=directory/'rirs/native_stage'/scene/f'{sample:05d}.npy')
            fig,axes=plt.subplots(3,1,figsize=(9,6),sharex=True)
            for name,path in refs.items():
                r=np.load(path)
                energy=m.band_energy(r,c).reshape(6,3,-1).sum(0)
                energy/=max(energy.sum(),1e-30)
                t=2+(np.arange(energy.shape[-1])*64+128)/48
                for band,ax in enumerate(axes):ax.plot(t,energy[band],label=name,linewidth=1.2)
            for label,ax in zip(('0–500 Hz','500–1500 Hz','1500–4000 Hz'),axes):
                ax.set_ylabel(label);ax.grid(alpha=.2)
            axes[0].legend(ncol=3);axes[-1].set_xlabel('Time after each RIR direct peak (ms)')
            axes[0].set_title(f'{scene} / Q{sample:05d} — GT pose; same 8 FP/native candidate poses')
            fig.tight_layout();fig.savefig(output/f'{key}.png',dpi=120);plt.close(fig)
            image=rgb_root/q['image_path']
            if image.exists():shutil.copy2(image,output/f'{key}_rgb.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','directory','source','output','rgb-root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.directory,a.source,a.output,a.rgb_root)
