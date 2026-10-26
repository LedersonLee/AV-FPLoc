"""Matched raw/common-removed observed/native/FP timing plots at GT poses."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import read
from .planar_probe import arrival_features
from .stage_comparison import stage_path


def run(root,report):
    plan=read(root/'AcousticContrastive/diagnostics/learning_failure_v1/geometry_control/plan.json')
    stage=root/'AcousticContrastive/diagnostics/stage_vs_extruded_v1'
    output=report/'common_figures';output.mkdir(exist_ok=True)
    t=2.25+np.arange(252)*.5
    for scene,queries in plan['scenes'].items():
        for q in queries:
            observed=q['observed'];positive=q['candidates'][0]
            paths=dict(furnished=root/observed['path'],native_stage=stage_path(stage,positive['path']),
                       extruded_FP=root/positive['path'])
            fig,axes=plt.subplots(6,2,figsize=(12,10),sharex=True,sharey=True)
            for name,path in paths.items():
                x=np.roll(np.load(path),observed['roll'],axis=0).astype(float)
                for column,array in enumerate((x,x-x.mean(0))):
                    envelope,_,_=arrival_features(array)
                    for mic in range(6):axes[mic,column].plot(t,envelope[mic],label=name,lw=1)
            axes[0,0].set_title('Raw RIR → normalized band envelope')
            axes[0,1].set_title('Subtract microphone mean → normalized band envelope')
            axes[0,0].legend(fontsize=7,ncol=3)
            for i in range(6):
                axes[i,0].set_ylabel(f'Mic {i}')
                for ax in axes[i]:ax.set_ylim(0,1.05);ax.set_xlim(2,128);ax.grid(alpha=.15)
            for ax in axes[-1]:ax.set_xlabel('Time after direct-sample origin (ms)')
            fig.suptitle(f'{scene} / Q{q["sample_index"]:05d} — all three RIRs at GT pose; no alignment fitting',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.97));fig.savefig(output/f'{scene}_{q["sample_index"]:05d}.png',dpi=100);plt.close(fig)
    page=report/'index.html';text=page.read_text()
    text=text.replace('<img id="plot"', '<h3>도면의 예상 도달 시간</h3><img id="plot"',1)
    text=text.replace('</section>\n<section><h2>자료',
        '<h3>같은 GT에서의 관측·Native stage·압출 FP: 공통 성분 제거 전/후</h3><p>왼쪽 원본, 오른쪽 마이크 평균 제거 후. 각 곡선은 개별 peak로 정규화한 500–1500Hz envelope입니다. 절대 에너지나 제거량 비교는 아니며, 시간 구조를 보기 위한 그림입니다.</p><img id="commonPlot" alt="Before/after common-mode removal, same GT pose" style="width:100%"></section>\n<section><h2>자료',1)
    text=text.replace("q.addEventListener('change',update);update()", "const originalUpdate=update;function updateAll(){originalUpdate();document.getElementById('commonPlot').src='common_figures/'+q.value+'.png'}q.addEventListener('change',updateAll);updateAll()")
    temp=page.with_name('index.common.tmp');temp.write_text(text);temp.replace(page)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();run(a.root,a.report)
