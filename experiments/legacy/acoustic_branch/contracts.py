"""Canonical HSSD acoustic contracts and atomic manifest utilities.

The raw-data protocol matches the immutable RIR manifests. Experiment render
workers remain untouched while their distributed jobs are running.
"""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

PROTOCOL = dict(sample_rate_hz=48000, indirect_rays=20000, ray_depth=50,
                source_rays=512, diffraction=True, max_diffraction_order=10,
                ring_radius_m=.05, materials=False, transmission=False)
RING = np.array([np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi, np.pi/3, 2*np.pi/3])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing immutable contract differs: {path}')
    else:
        atomic_json(path, value)


def recipe(yaw):
    acoustic = (-float(yaw)) % 360
    base = acoustic % 60
    if not np.isclose(base, round(base)):
        raise ValueError('Only integer acoustic headings supported')
    return int(round(base)), -int(round((acoustic-base)/60))
