"""Copy only the task's small report outputs to the existing LAN report server."""
import argparse
from pathlib import Path
import shutil
import time


def run(source,dest):
    import json
    source,dest=Path(source),Path(dest);dest.mkdir(parents=True,exist_ok=True)
    while True:
        # Publish HTML last so referenced result files precede the new page.
        for name in ('summary.json','records.json','metrics.csv','status.json','index.html'):
            path=source/name
            if path.exists():
                tmp=dest/(name+'.pending');shutil.copy2(path,tmp);tmp.replace(dest/name)
        status=source/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status') in ('complete','failed'):return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--dest',required=True)
    a=p.parse_args();run(a.source,a.dest)
