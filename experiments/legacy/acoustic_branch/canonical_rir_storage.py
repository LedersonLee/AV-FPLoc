"""Canonical raw-data layout migration; no array rewriting or resampling."""
import argparse
from pathlib import Path
import time
import json
import subprocess

from .contracts import atomic_json, immutable_json

ROOT=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD')
OLD='RIR/FP_corrected_yup_v2'
CANONICAL='RIR/FP'
MANIFEST='RIR/manifests/storage_layout_v1.json'


def migrate(root):
    if root!=ROOT or root.is_symlink():
        raise ValueError('Only the verified File2 HSSD dataset is in scope')
    old=root/OLD
    canonical=root/CANONICAL
    archive=root/'RIR/Archive/invalid_loader_metadata'
    rows=[]
    for split in ('train','val','test'):
        source=old/split
        destination=canonical/split
        saved=archive/split
        if not source.is_dir() or source.is_symlink():
            raise ValueError('Expected exact corrected source directory: '+str(source))
        if saved.exists():
            raise FileExistsError('Refuse to overwrite prior metadata archive: '+str(saved))
        rows.append((source,destination,saved,source.stat().st_ino))
    # Callers must stop all FP writers/readers with physical-path bind mounts
    # first. All moves are same-filesystem atomic directory renames.
    archive.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'RIR/manifests/storage_migration_progress.json',dict(status='running',updated_unix=time.time()))
    for source,destination,saved,inode in rows:
        if destination.exists():
            destination.rename(saved)
        source.rename(destination)
        if destination.stat().st_ino!=inode:
            raise RuntimeError('Directory identity changed during migration')
    old.rmdir()  # Empty directory only; never recursively delete dataset data.
    immutable_json(root/MANIFEST,dict(status='complete',layout='canonical_rir_storage_v1',
        fp_prefix=CANONICAL,observed_prefix='RIR/Observed',
        previous_logical_fp_prefix=OLD,
        splits=['train','val','test'],raw_arrays_rewritten=False,
        invalid_legacy_metadata_archive='RIR/Archive/invalid_loader_metadata',
        checkpoint_configs_and_raw_hashes_unchanged=True,
        convention='Versions and corrections belong in manifests, not raw-data folder names'))
    atomic_json(root/'RIR/manifests/storage_migration_progress.json',dict(status='complete',updated_unix=time.time()))
    print('Canonical FP data moved to '+str(canonical),flush=True)


def resume(root):
    from .launch_node2_preprocess import remote,SSH
    out=root/'AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2/evaluation'
    native=json.loads((out/'bank_launch.json').read_text())['command'].copy()
    native_name='hssd-canonical-test-node2-20261019-v1'
    native[native.index('--name')+1]=native_name
    for i in reversed(range(1,len(native))):
        if native[i-1]=='-v' and ':/datasets/HSSD/RIR/FP_corrected_yup_v2:rw' in native[i]:
            del native[i-1:i+1]
    native=[v.replace(str(root/OLD/'test')+':',str(root/CANONICAL/'test')+':') for v in native]
    local=json.loads((out/'evaluation_launch.json').read_text())['command'].copy()
    local_name='hssd-canonical-evaluation-local-20261019-v1'
    local[local.index('--name')+1]=local_name
    receipt=root/'RIR/manifests/storage_migration_runtime.json'
    if receipt.exists():
        raise FileExistsError('Runtime migration already launched; inspect its receipt before recovery')
    native_id=remote(native).strip()
    local_id=subprocess.check_output(local,text=True).strip()
    immutable_json(receipt,dict(native_container_id=native_id,local_container_id=local_id,
        native_command=native,local_command=local,raw_data_prefix=CANONICAL,
        checkpoint_and_features_unchanged=True,
        note='Reuses exact immutable renderer/evaluator code; canonical physical mount replaces old prefix'))
    print(dict(native=native_id,evaluator=local_id),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.resume:resume(a.root)
    else:migrate(a.root)
