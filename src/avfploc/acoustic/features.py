"""Official 48-kHz RIR features; fixed clock and ordered six-microphone ring.

No torch dependency: feature extraction/caching can run on CPU render hosts.
"""
import hashlib
import json

import numpy as np
from scipy import signal


DEFAULT_FEATURES = dict(version='energy_complex_v1', sample_rate_hz=48000,
    direct_sample=7, start_ms=2, end_ms=128, early_end_ms=50,
    nfft=256, hop=64, band_hz=[500, 4000], log_energy_gain=100.0)


def feature_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def validate_config(config):
    if set(config) != set(DEFAULT_FEATURES):
        raise ValueError('Feature config keys differ from the versioned schema')
    if config['version'] not in (DEFAULT_FEATURES['version'], 'energy_complex_zero_tail_v2') or config['sample_rate_hz'] != 48000:
        raise ValueError('Only canonical 48-kHz features supported')
    if config['direct_sample'] != 7:
        raise ValueError('Do not shift channels/records to their own peak')
    if not 0 <= config['start_ms'] < config['early_end_ms'] < config['end_ms']:
        raise ValueError('Invalid reflection windows')
    if not 0 <= config['band_hz'][0] < config['band_hz'][1] <= 24000:
        raise ValueError('Invalid frequency band')
    if config['nfft'] < 32 or not 0 < config['hop'] <= config['nfft'] or config['log_energy_gain'] <= 0:
        raise ValueError('Invalid STFT/energy parameters')


def extract(rir, config=DEFAULT_FEATURES):
    """energy[6,F,T], coherence[2 windows,6 adjacent pairs,2 Re/Im,F].

    Complex coherency is <Zi conj(Zi+1)> / sqrt(<|Zi|²><|Zi+1|²>).
    It is calculated WITHIN each RIR, not across observed and FP signals.
    A single energy normalization preserves relative mic/frequency/time energy.
    """
    validate_config(config)
    rir = np.asarray(rir)
    if rir.ndim != 2 or rir.shape[0] != 6 or not np.isfinite(rir).all():
        raise ValueError('Expected finite six-channel RIR')
    fs = config['sample_rate_hz']
    first = 7 + round(config['start_ms'] * fs / 1000)
    last = 7 + round(config['end_ms'] * fs / 1000)
    if not np.all(np.any(rir != 0, axis=-1)):
        raise ValueError('Silent RIR')
    if rir.shape[-1] < last and config['version'] == 'energy_complex_zero_tail_v2':
        if rir.shape[-1] < first + config['nfft']:
            raise ValueError('RIR does not cover the start of the reflection window')
        # RLR emits variable-length finite responses. Preserve all returned
        # samples, and represent the unreturned tail as zeros for fixed-size
        # STFT tensors. No peak shifting, interpolation, or synthetic echoes.
        rir = np.pad(rir, ((0, 0), (0, last-rir.shape[-1])))
    if rir.shape[-1] < last:
        raise ValueError('Short/silent RIR; no padding or synthetic repair allowed')
    f, t, z = signal.stft(rir[:, first:last].astype(np.float64), fs=fs,
        nperseg=config['nfft'], noverlap=config['nfft'] - config['hop'],
        boundary='zeros', padded=True, axis=-1)
    times = config['start_ms'] + t * 1000
    keep = (f >= config['band_hz'][0]) & (f < config['band_hz'][1]) & (f > 0) & (f < fs / 2)
    z = z[:, keep, :][..., times < config['end_ms']]
    times = times[times < config['end_ms']]
    if z.shape[1] < 2 or not np.any(times < config['early_end_ms']):
        raise ValueError('Insufficient STFT bins/frames')
    energy = abs(z) ** 2
    mass = energy.sum()
    if not np.isfinite(mass) or mass <= np.finfo(np.float64).tiny:
        raise ValueError('No reflection energy in selected band/window')
    energy = np.log1p(config['log_energy_gain'] * energy / mass)
    coherence = []
    for mask in (times < config['early_end_ms'], np.ones(len(times), dtype=bool)):
        a = z[..., mask]
        b = np.roll(a, -1, axis=0)
        cross = np.mean(a * b.conj(), axis=-1)
        denom = np.sqrt(np.mean(abs(a) ** 2, axis=-1) * np.mean(abs(b) ** 2, axis=-1))
        c = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-30)
        coherence.append(np.stack([c.real, c.imag], axis=1))
    return energy.astype(np.float32), np.asarray(coherence, dtype=np.float32)


def roll_features(energy, coherence, roll):
    # Rolling channels also rolls the ordered adjacent-pair list; no conjugation.
    return np.roll(energy, int(roll), axis=0), np.roll(coherence, int(roll), axis=1)
