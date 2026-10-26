"""Publish complete, portable pilot results; no external assets or libraries."""
import shutil
from pathlib import Path


def build_report(root,report,records,banks,summary):
    (report/'rgb').mkdir(exist_ok=True);(report/'maps').mkdir(exist_ok=True)
    from .data import safe_path
    for scene in banks:
        shutil.copyfile(root/'2DFP'/scene/'map.png',report/'maps'/f'{scene}.png')
    for r in records:
        image=safe_path(root,r['image_path'])
        name=f'{r["scene_id"]}_{r["sample_index"]}.png'
        if not (report/'rgb'/name).exists():shutil.copyfile(image,report/'rgb'/name)
    shutil.copyfile(Path(__file__).with_name('templates')/'pilot_report.html',report/'index.html')
    lines=['# HSSD acoustic pilot: same-bank results','',
        'Exploratory 3-scene comparison; vision116-scene training vs acoustic12. No test fitting.',
        '', '| Method | R@0.1m | R@0.5m | R@1m | R@1m/30deg | Median XY(m) | repair | harm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,m in summary['micro'].items():
        lines.append(f'| {method} | {100*m["recall_0.1m"]:.2f}% | {100*m["recall_0.5m"]:.2f}% | '
                     f'{100*m["recall_1m"]:.2f}% | {100*m["recall_1m_30deg"]:.2f}% | '
                     f'{m["median_xy_m"]:.2f} | {m["repair"]} | {m["harm"]} |')
    lines += ['', '## Observed results and failure analysis',''] + ['- '+s for s in summary.get('interpretations',[])]
    lines += ['', '## Failure interpretation','',
        '- Top-K exclusion: the visual proposal pool has no candidate within1m; acoustics cannot repair this pool.',
        '- Acoustic selection failure: Top-K includes a candidate within1m but the reranker chooses another location.',
        '- Yaw-only failure: location within1m, heading outside30deg. The primary top-K rule retains visual heading.',
        '- Geometry QA flags are existing review signals, not a confirmed cause. Inspect RGB/map/depth together.',
        '- Pose-clustered paired CIs cover only these exploratory scenes, not unseen-scene generalization.',
        '', 'Detailed per-scene/K metrics, exclusions, input hashes and intervals: summary.json.',
        'Per-query outcomes, diagnostic flags and scores: records.json / heatmaps/.']
    (report/'results.md').write_text('\n'.join(lines)+'\n')
