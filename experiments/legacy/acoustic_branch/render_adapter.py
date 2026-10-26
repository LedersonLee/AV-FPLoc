"""Optional bridge to the existing native RIR generation infrastructure.

Only prepare-val/render-val need this adapter. Training, feature extraction,
verification and checkpoint loading do not import the legacy renderer. Avoid
moving live render workers in the middle of a distributed generation run.
"""
import importlib
from pathlib import Path
import sys


def _tools(name):
    directory = Path(__file__).resolve().parents[1] / 'experiments/exp4_acoustic_xy_observability/scripts'
    if not directory.is_dir():
        raise RuntimeError('Optional native RIR tools unavailable; use prepared RIR schedules instead')
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(directory))


def positive_tools():
    return _tools('hssd_positive_pairs')


def supervisor_tools():
    return _tools('hssd_rir_supervisor')
