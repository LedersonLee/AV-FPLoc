"""Standalone experimental report for planar first-order reflection probes."""
import argparse
import csv
import html
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .contracts import atomic_json
from .data import read
from .planar_probe import arrival_features,proxy_path


def run(a):
    data=read(a.source/'results.json');a.output.mkdir(parents=True,exist_ok=True)
    rows=[dict(geometry=g,method=m,**v['summary']) for g,methods in data['methods'].items() for m,v in methods.items()]
    with (a.output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copy2(a.source/'results.json',a.output/'results.json')
    shutil.copy2(a.source/'contract.json',a.output/'contract.json')
    table=''.join(f'<tr><td>{r["geometry"]}</td><td>{r["method"]}</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in ('sampled_top1','recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg'))+f'<td>{r["median_xy_error_m"]:.2f}m</td><td>{r["tied_queries"]}</td><td>{100*r["gt_event_coverage"]:.1f}%</td></tr>' for r in rows)
    control=read(a.root/'AcousticContrastive/diagnostics/learning_failure_v1/geometry_control/plan.json')
    figures=a.output/'figures';figures.mkdir(exist_ok=True)
    options=[]
    for scene,queries in control['scenes'].items():
        for q in queries:
            key=f'{scene}_{q["sample_index"]:05d}';options.append(f'<option value="{key}">{scene} / Q{q["sample_index"]:05d}</option>')
            ref=q['candidates'][0];roll=q['observed']['roll']
            rir=np.load(a.root/q['observed']['path'])
            env,events,_=arrival_features(np.roll(rir,roll,axis=0))
            path=proxy_path(a.raw_proxy or a.source,'planar_first',ref['path'])
            meta=read(path.with_suffix('.json'))
            paths=np.roll(np.array(meta['paths'],dtype=object),roll)
            fig,axes=plt.subplots(6,1,figsize=(10,8),sharex=True)
            t=2.25+np.arange(252)*.5
            for i,ax in enumerate(axes):
                ax.plot(t,env[i],color='#2463b4',label='Observed 500–1500 Hz envelope')
                for j,length in enumerate(paths[i]['horizontal_lengths_m']):
                    ax.axvline(length/343*1000-7/48,color='#d77513',alpha=.8,label='2D first wall arrival' if j==0 else None)
                for j,length in enumerate(paths[i]['vertical_lengths_m']):
                    ax.axvline(length/343*1000-7/48,color='#4c9f8a',linestyle=':',label='Floor/ceiling control' if j==0 else None)
                ax.set_ylim(0,1.05);ax.set_ylabel(f'Mic{i}');ax.grid(alpha=.15)
            axes[0].legend(loc='upper right',fontsize=7,ncol=3)
            axes[-1].set_xlabel('Time after direct-sample origin (ms); NO per-record alignment')
            axes[-1].set_xlim(2,128)
            axes[0].set_title(f'{scene} / Q{q["sample_index"]:05d} — GT pose, diagnostic only')
            fig.tight_layout();fig.savefig(figures/f'{key}.png',dpi=110);plt.close(fig)
            rgb=a.rgb_root/q['image_path']
            if rgb.exists():shutil.copy2(rgb,figures/f'{key}_rgb.png')
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Planar first-order acoustic probe</title><style>body{{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#f4f7fb;color:#17243a}}section{{background:white;border:1px solid #ddd;border-radius:10px;padding:20px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}.warning{{background:#fff0d2;padding:16px}}</style><h1>도면에 남는 수평 반사 단서 — 1차 경로 실험</h1>
<p class="warning">Val 3 scenes / 1,704 queries. 학습·test 없음. planar_first는 수평 벽의 1차 specular 경로 proxy이며, 고차 잔향·회절을 포함한 완전한 2D RIR 시뮬레이터가 아닙니다. 3D spreading을 수평 경로에 적용했고 엄밀한 cylindrical-wave 모델은 아닙니다. RLR와의 차이를 차원 효과 하나로 해석하면 안 됩니다.</p>
<section><h2>전체 결과</h2><p>native_RLR / extruded_RLR: 기존 고차 RIR. planar_first: 바닥·천장 없는 수평 1차 경로. vertical_first: 동일한 계산·벽 경로에 바닥·천장 1차 반사 두 개만 추가한 대조군. 후보 pose/yaw, 관측 RIR은 동일합니다. 정답이 포함된 촬영 pose bank이지 dense grid 결과가 아닙니다.</p><div class="scroll"><table><tr><th>후보 모델</th><th>비교 방식</th><th>8후보 Top1</th><th>R@0.1m</th><th>R@0.5m</th><th>R@1m</th><th>R@1m &amp;30°</th><th>위치 중앙값</th><th>동점 쿼리</th><th>GT 이벤트 coverage</th></tr>{table}</table></div></section>
<section><h2>매칭 규약</h2><p>STFT_L1/coherence_L2는 이전 비교와 같은 규약입니다. arrival_L1은 500–1500Hz의 6마이크별 정규화 envelope를 고정 시계로 비교합니다. arrival_partial은 후보에서 예상한 peak들이 관측 envelope에서 지지되는 정도를 평균하며 관측의 추가 peak 자체는 벌점 처리하지 않습니다. 0.5ms bin, ±0.5ms 허용, direct+2ms 이전 제거 후 filtering, 시간축 이동 최적화 없음. 빈 후보는 최악 점수이며, exact tie는 GT와 무관한 고정 hash 순서로 해소했습니다. Coverage가 높아도 모든 후보가 비슷하면 위치 구분력은 낮을 수 있습니다.</p><p>common_* 비교가 포함된 경우: 각 시각의 6마이크 평균을 빼 공통 성분을 제거했습니다. 중앙 음원·같은 높이의 수평 ring이라는 현재 배치에 한정된 가정이며, 불명 위치의 수동 음원에는 그대로 적용할 수 없습니다. early_*는 시작 0.35ms / 첫 구간 끝20ms, 그 외 coherence는 시작2ms / 첫 구간 끝50ms이고 전체 구간 끝128ms는 같습니다. 시간 정렬을 GT에 맞춰 최적화하지 않습니다.</p></section>
<section><h2>GT에서 수평 반사 도달 시간이 실제로 보이는가?</h2><select id="q">{''.join(options)}</select><p>파랑: 관측 low-band envelope. 주황: 도면의 수평 1차 반사 시간. 녹색 점선: 바닥·천장 반사 대조군. 관측을 GT에 맞춰 시간 이동시키지 않았습니다. GT 표시는 진단용이며 후보 선택에는 GT를 넣지 않았습니다.</p><img id="rgb" alt="Query RGB" style="width:45%"><img id="plot" alt="Per-microphone observed envelope and predicted first-wall arrival times" style="width:100%"></section>
<section><h2>자료</h2><p><a href="metrics.csv">CSV</a> · <a href="results.json">Scene별·쿼리별 전체 결과</a> · <a href="contract.json">가정·한계·입력/코드 SHA</a> · <a href="../stage-comparison/">3D stage vs 압출 FP 비교</a></p><p>이 결과로 특정 설정을 test에 적용하기 전 별도 확인이 필요합니다. 여러 val 설정을 탐색했으며, 개선된 설정만 골라 독립적인 최종 성능인 것처럼 보고하지 않습니다. map·기존 RIR·scene split은 변경하지 않았습니다.</p></section><script>const q=document.getElementById('q');function update(){{document.getElementById('rgb').src='figures/'+q.value+'_rgb.png';document.getElementById('plot').src='figures/'+q.value+'.png'}}q.addEventListener('change',update);update()</script></html>'''
    (a.output/'index.html').write_text(text)
    atomic_json(a.source/'report.json',dict(status='complete',path=str(a.output)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','root','rgb-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--raw-proxy',type=Path)
    run(p.parse_args())
