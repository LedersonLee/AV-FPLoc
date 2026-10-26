import numpy as np
from scipy.special import softmax
from avfploc.evaluation.baselines import standardized


def extract(vision, acoustic, xy, depth, schema="core14", ray_bank=None):
    """Versioned historical features; scores must already be full-bank z-scored.

    No GT input. Preserve old 14/17/58 column order, including MAP-envelope
    boundary conventions. Degenerate inputs fail rather than silently impute.
    """
    depth = np.asarray(depth)
    base = features(vision, acoustic, np.asarray(xy), depth)
    core = base[[*range(10), 13, 14, 15, 16]]
    if schema == "legacy17":
        return base, FEATURE_NAMES
    if schema == "core14":
        return core, CORE_NAMES
    if schema != "processed58" or ray_bank is None:
        raise ValueError("processed58 requires candidate DESDF rays [XY,36]")
    if np.asarray(ray_bank).shape != np.asarray(vision).shape:
        raise ValueError("Ray bank candidate/yaw mismatch")
    extra, names, _ = engineered(vision, acoustic, np.asarray(xy), depth,
                                 np.asarray(ray_bank), envelope(vision, acoustic))
    result = np.r_[core, extra]
    if len(result) != 58:
        raise ValueError("Unexpected processed58 schema")
    return result, CORE_NAMES + names

def spatial_stats(score,xy):
    """Fixed-temperature descriptive distributions, NOT calibrated confidence.

    XY uses max over yaw then zscore+softmax; normalized entropy comparable in
    support size, not proven calibrated across scenes. Repeat at T=0.5,2.
    """
    s=score.max(1).astype(float)
    z=standardized(s)
    top=np.argsort(-s,kind='stable')[:50]
    peak=int(top[0])
    dist=np.linalg.norm(xy-xy[peak],axis=1)
    separated=dist>1
    probability=softmax(z)
    entropy=lambda p:float(-(p*np.log(np.maximum(p,1e-300))).sum()/np.log(len(p)))
    modes=[]
    for i in top:
        if not modes or np.min(np.linalg.norm(xy[modes]-xy[i],axis=1))>1:
            modes.append(int(i))
    yaw_probability=softmax(standardized(score[peak]))
    return dict(entropy=entropy(probability),entropy_t05=entropy(softmax(z/.5)),
        entropy_t2=entropy(softmax(z/2)),peak_mass_1m=float(probability[~separated].sum()),
        separated_gap=float(z[peak]-z[separated].max()) if separated.any() else 0.,
        top50_modes=len(modes),top50_rms_m=float(np.sqrt(np.mean(np.sum((xy[top]-xy[top].mean(0))**2,axis=1)))),
        yaw_entropy=entropy(yaw_probability)),probability,top

FEATURE_NAMES=[*(f'{m}_{f}' for m in ('vision','acoustic') for f in
    ('separated_gap','spread_over_diagonal','peak_mass_1m','xy_entropy','yaw_entropy')),
    'visual_depth_mean','visual_depth_cv','visual_depth_range',
    'map_distance_over_diagonal','yaw_disagreement_fraction','xy_overlap','top50_overlap']


def features(vision,acoustic,xy,depth):
    """No GT, scene ID or absolute coordinate enters the learned gate."""
    xy=np.asarray(xy);depth=np.asarray(depth)
    diagonal=max(float(np.linalg.norm(np.ptp(xy,axis=0))),1.)
    values=[];stats=[]
    for score in (vision,acoustic):
        st,prob,top=spatial_stats(score,xy);stats.append((st,prob,top))
        values.extend([st['separated_gap'],st['top50_rms_m']/diagonal,
            st['peak_mass_1m'],st['entropy'],st['yaw_entropy']])
    values.extend([float(depth.mean()),float(depth.std()/max(abs(depth.mean()),1e-3)),float(np.ptp(depth))])
    vi,ai=[np.unravel_index(s.argmax(),s.shape) for s in (vision,acoustic)]
    values.extend([float(np.linalg.norm(xy[vi[0]]-xy[ai[0]])/diagonal),
        float(abs(((vi[1]-ai[1])*10+180)%360-180)/180),
        float(np.sqrt(stats[0][1]*stats[1][1]).sum()),
        len(set(stats[0][2])&set(stats[1][2]))/min(50,len(xy))])
    values=np.asarray(values,np.float32)
    if values.shape!=(len(FEATURE_NAMES),) or not np.isfinite(values).all():raise ValueError('Invalid gate feature vector')
    return values

CORE_NAMES = [FEATURE_NAMES[i] for i in [*range(10),13,14,15,16]]
RESPONSE_WEIGHTS = [.05,.15,.2,.3,.5]
def envelope(v, a):
    """All positive-width MAP-winner intervals of (1-w)*v+w*a, float64.

    Divide at intersections of endpoint winners. If another line lies above
    their intersection, recurse; otherwise the pair forms the upper envelope.
    Isolated tie ownership at endpoints is intentionally not an interval.
    """
    v = np.asarray(v, np.float64).ravel()
    d = np.asarray(a, np.float64).ravel() - v
    parts = []

    def visit(p, q, lo, hi, depth=0):
        if depth > 500:
            raise ValueError('Envelope recursion failed')
        if p == q:
            parts.append([lo, hi, int(p)])
            return
        den = d[q] - d[p]
        if den <= 0:
            raise ValueError('Endpoint slope order failed')
        t = float((v[p] - v[q]) / den)
        if not lo - 1e-9 <= t <= hi + 1e-9:
            raise ValueError('Intersection outside bracket')
        t = float(np.clip(t, lo, hi))
        scores = v + t * d
        r = int(scores.argmax())
        tol = 1e-11 * max(1., abs(float(scores[r])))
        if scores[r] - scores[p] <= tol:
            parts.extend([[lo, t, int(p)], [t, hi, int(q)]])
        else:
            visit(p, r, lo, t, depth + 1)
            visit(r, q, t, hi, depth + 1)

    visit(int(v.argmax()), int(np.asarray(a).argmax()), 0., 1.)
    result = []
    for lo, hi, idx in parts:
        if hi - lo <= 1e-13:
            continue
        if result and result[-1][2] == idx:
            result[-1][1] = hi
        else:
            result.append([lo, hi, idx])
    if abs(sum(hi - lo for lo, hi, _ in result) - 1) > 1e-8:
        raise ValueError('Incomplete envelope')
    return result

def winner_at(parts, w):
    for lo, hi, idx in parts:
        if lo - 1e-12 <= w <= hi + 1e-12:
            return int(idx)
    raise ValueError('Weight outside envelope')

def pose(idx, xy):
    return np.r_[xy[idx // 36], idx % 36 * 10.]

def yawdiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.) % 360. - 180.)

def engineered(v, a, xy, depth, ray_bank, parts):
    """NO GT, success labels, scene IDs, absolute coordinates or Laplace scale.

    Relative geometric distances are normalized by map diagonal. Residuals use
    UnLoc predicted depth and DESDF only, not captured/furnished/GT ray depth.
    """
    diag = max(float(np.linalg.norm(np.ptp(xy, axis=0))), 1.)
    fields = [np.asarray(v), np.asarray(a)]
    ids = [int(s.argmax()) for s in fields]
    points = [pose(i, xy) for i in ids]
    names, values, families = [], [], []

    def add(name, val, family):
        names.append(name); values.append(float(val)); families.append(family)

    angles = np.deg2rad(np.arange(-5, 6) * 10.)
    coords = np.tan(angles) * len(depth) * 3 / 8 + (len(depth) - 1) / 2
    rays = np.interp(coords, np.arange(len(depth)), depth) / np.cos(angles)
    for name, val in [('mean', depth.mean()), ('cv', depth.std() / max(abs(depth.mean()), 1e-3)),
                      ('range', np.ptp(depth)), ('adjacent_variation', np.abs(np.diff(depth)).mean())]:
        add('depth_' + name, val, 'depth')
    for k, field in enumerate(fields):
        label = ('vision', 'acoustic')[k]
        own, other = ids[k], ids[1-k]
        flat = field.ravel(); s = field.max(1); other_cell = other // 36
        add(label + '_other_pose_rank_fraction', np.mean(flat >= flat[other]), 'support')
        add(label + '_other_xy_rank_fraction', np.mean(s >= s[other_cell]), 'support')
        add(label + '_other_pose_score_deficit', flat[own] - flat[other], 'support')
        add(label + '_other_xy_score_deficit', s[own // 36] - s[other_cell], 'support')
        dist = np.linalg.norm(xy - points[1-k][:2], axis=1)
        z = (s - s.mean()) / max(s.std(), 1e-8)
        add(label + '_mass_at_other_1m', softmax(z)[dist <= 1].sum(), 'support')
        top = np.argsort(-s, kind='stable')[:50]; modes = []
        for idx in top:
            if not modes or np.min(np.linalg.norm(xy[modes] - xy[idx], axis=1)) > 1:
                modes.append(int(idx))
        add(label + '_top50_spatial_modes', len(modes), 'support')
    vs, acs = v.max(1), a.max(1)
    add('xy_score_correlation', np.corrcoef(vs, acs)[0, 1], 'support')
    topn = max(1, int(np.ceil(len(xy) * .01)))
    vt = np.argpartition(-vs, topn - 1)[:topn]; at = np.argpartition(-acs, topn - 1)[:topn]
    add('top1percent_xy_overlap', len(set(vt) & set(at)) / topn, 'support')
    base = pose(winner_at(parts, .1), xy)
    for w in RESPONSE_WEIGHTS:
        p = pose(winner_at(parts, w), xy)
        add(f'w{w:g}_move_over_diagonal', np.linalg.norm(p[:2] - base[:2]) / diag, 'response')
        add(f'w{w:g}_yaw_move_fraction', yawdiff(p[2], base[2]) / 180., 'response')
    selected = next((lo, hi) for lo, hi, idx in parts if lo <= .1 <= hi)
    add('fixed01_winner_interval_width', selected[1] - selected[0], 'response')
    add('fixed01_stability_left', .1 - selected[0], 'response')
    add('fixed01_stability_right', selected[1] - .1, 'response')
    add('winner_transitions_to05', sum(lo > .1 and lo < .5 for lo, _, _ in parts), 'response')
    bearings = (np.arange(36)[:, None] + np.arange(-5, 6)) % 36
    for name, pidx in [('vision', ids[0]), ('acoustic', ids[1]), ('fixed01', winner_at(parts, .1))]:
        pred = ray_bank[pidx // 36, bearings[pidx % 36]]
        delta = np.abs(pred - rays)
        add(name + '_ray_mae_m', delta.mean(), 'ray_fit')
        add(name + '_ray_median_ae_m', np.median(delta), 'ray_fit')
        add(name + '_ray_fraction_within05m', (delta <= .5).mean(), 'ray_fit')
        add(name + '_ray_relative_mae', delta.mean() / max(float(np.mean(np.abs(rays))), .1), 'ray_fit')
    result = np.asarray(values, np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite GT-free features')
    return result, names, families
