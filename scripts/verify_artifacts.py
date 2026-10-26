"""Read-only shared artifact preflight; never regenerates or modifies data."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=os.environ.get('AVFPLOC_DATA_ROOT'))
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'configs/datasets/hssd_release.json')
    p.add_argument('--full-hash',action='store_true');a=p.parse_args()
    if a.root is None:p.error('--root or AVFPLOC_DATA_ROOT required')
    root=a.root.resolve();manifest=json.loads(a.manifest.read_text());fail=[]
    for ref in manifest['files']:
        path=(root/ref['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():fail.append(ref['path']+': missing/outside root');continue
        if path.stat().st_size!=ref['size']:fail.append(ref['path']+': size mismatch');continue
        if a.full_hash:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=ref['sha256']:fail.append(ref['path']+': SHA256 mismatch')
    print(json.dumps(dict(ok=not fail,checked=len(manifest['files']),full_hash=a.full_hash,errors=fail),indent=2))
    raise SystemExit(1 if fail else 0)


if __name__=='__main__':main()
