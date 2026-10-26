"""Publish the existing14 feature-importance diagnostic, without Test access."""
import argparse
import csv
from pathlib import Path
import shutil

from .core14_importance import REL, CORE_NAMES, read, save


def run(args):
    from acoustic_branch.contracts import sha
    out = args.root/REL; data = read(out/'results.json'); report = args.report
    assert data['status'] == 'complete' and data['test_accessed'] is False
    report.mkdir(parents=True, exist_ok=True)
    for name in ('results.json', 'correlations.json', 'config.json'):
        shutil.copy2(out/name, report/name)
    shutil.copy2(Path(__file__).with_name('core14_importance_report.html'), report/'index.html')
    rows = []
    for task in data['classifiers']:
        for r in task['variants']:
            for split in ('train_oof', 'val'):
                rows.append(dict(model=task['model'], criterion=task['criterion'], action_weight=task['action_weight'],
                                 variant=r['variant'], inputs=r['input_count'], split=split, **r[split]))
    with (out/'classifier_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    shutil.copy2(out/'classifier_metrics.csv', report/'classifier_metrics.csv')
    ver = dict(status='passed', **data['verification'], test_accessed=False,
               train_scenes=12, val_scenes=3, classifier_refits=8*35*5,
               feature_definitions=CORE_NAMES, report_files={n: sha(report/n) for n in ('results.json', 'index.html', 'classifier_metrics.csv')},
               source_code={n: sha(Path(__file__).with_name(n)) for n in ('core14_importance.py', 'core14_importance_report.py', 'core14_importance_report.html')})
    save(out/'verification.json', ver); save(report/'verification.json', ver)
    print('REPORT', report)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    run(p.parse_args())
