"""Fetch pinned upstream sources and apply our tracked patch; never overwrite clones."""
import argparse
from pathlib import Path
import subprocess

UNLOC='3c7a301aa08049f6b1bb86f110e8e6e675e2300b'
DEPTH='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'


def run(*args,cwd=None):subprocess.run(args,cwd=cwd,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    dest=a.destination.resolve()
    if dest.exists():raise FileExistsError('Destination exists; choose an empty new path')
    run('git','clone','https://github.com/matthias-wueest/UnLoc.git',str(dest))
    run('git','checkout','--detach',UNLOC,cwd=dest)
    run('git','submodule','update','--init','--recursive',cwd=dest)
    run('git','checkout','--detach',DEPTH,cwd=dest/'depth_anything_v2')
    patch=Path(__file__).resolve().parents[1]/'external/patches/unloc-hssd.patch'
    run('git','apply','--check',str(patch),cwd=dest)
    run('git','apply',str(patch),cwd=dest)
    print('Pinned sources ready. Supply licensed/downloaded DA weights separately; see docs/hssd_reproduction.md')


if __name__=='__main__':main()
