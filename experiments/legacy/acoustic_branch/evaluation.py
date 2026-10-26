"""Same-bank pilot comparison. GT is used ONLY by diagnostics/metrics.

Scores are higher-is-better [XY cells, yaw bins]. Equal-weight fusion uses
per-query full-bank z scores, not equal raw numbers with unrelated scales.
"""
import numpy as np


METHODS = ('vision_only', 'acoustic_only', 'equal_weight_fusion', 'visual_topk_acoustic_rerank')


def standardized(score):
    score = np.asarray(score, dtype=np.float64)
    std = score.std()
    return (score-score.mean())/std if std > 1e-12 else np.zeros_like(score)


def select(vision, acoustic, k=50):
    """No GT argument and no GT-injected candidates. Stable bank-order ties."""
    vision, acoustic = np.asarray(vision), np.asarray(acoustic)
    if vision.shape != acoustic.shape or vision.ndim != 2 or not vision.size:
        raise ValueError('All methods require the same nonempty XY/yaw bank')
    if not np.isfinite(vision).all() or not np.isfinite(acoustic).all() or k < 1:
        raise ValueError('Invalid score bank/K')
    def maximum(score):
        return tuple(int(v) for v in np.unravel_index(score.argmax(),score.shape))
    chosen = dict(vision_only=maximum(vision), acoustic_only=maximum(acoustic),
        equal_weight_fusion=maximum(.5*standardized(vision)+.5*standardized(acoustic)))
    top = np.argsort(-vision.max(axis=1), kind='stable')[:k]
    cell = int(top[acoustic[top].max(axis=1).argmax()])
    chosen['visual_topk_acoustic_rerank'] = (cell,int(vision[cell].argmax()))
    return chosen, top


def diagnose(vision, acoustic, xy, truth, k=50, yaw_step=10):
    chosen, top = select(vision,acoustic,k)
    xy, truth = np.asarray(xy),np.asarray(truth)
    if xy.shape != (vision.shape[0],2) or len(truth) != 3 or vision.shape[1]*yaw_step != 360:
        raise ValueError('Candidate geometry/yaw shape mismatch')
    distance = np.linalg.norm(xy-truth[:2],axis=1)
    yaw = abs((np.arange(vision.shape[1])*yaw_step-truth[2]+180)%360-180)
    results = {}
    for method,(cell,heading) in chosen.items():
        results[method] = dict(cell=cell,yaw_bin=heading,xy_error_m=float(distance[cell]),
            yaw_error_deg=float(yaw[heading]),pose_xzyaw=[*xy[cell].tolist(),heading*yaw_step])
    vision_ok = results['vision_only']['xy_error_m'] <= 1
    for method in METHODS:
        ok = results[method]['xy_error_m'] <= 1
        results[method]['vs_vision_1m'] = ('preserve_success' if ok else 'preserve_failure') if ok == vision_ok else ('repair' if ok else 'harm')
    # Joint oracle over all allowed yaw bins is distinct from outputting the
    # selected cell's visual yaw. GT never participates in the selection above.
    oracle = {f'topk_oracle_{threshold:g}m': bool((distance[top] <= threshold).any()) for threshold in (.1,.5,1)}
    oracle['full_bank_oracle_1m'] = bool((distance<=1).any())
    oracle['topk_oracle_1m_30deg'] = bool((distance[top]<=1).any() and (yaw<=30).any())
    oracle['topk_with_visual_yaw_oracle_1m_30deg'] = bool(((distance[top]<=1)&(yaw[vision[top].argmax(axis=1)]<=30)).any())
    nearest = int(distance.argmin())
    audio_xy = acoustic.max(axis=1)
    oracle['gt_nearest_xy_audio_rank'] = 1+int((audio_xy>audio_xy[nearest]).sum())
    oracle['gt_nearest_xy_quantization_error_m'] = float(distance[nearest])
    return dict(methods=results,diagnostics=oracle)


def summarize(records):
    if not records:
        raise ValueError('Empty evaluation')
    result = {}
    for method in METHODS:
        values = [r['methods'][method] for r in records]
        xy = np.array([r['xy_error_m'] for r in values])
        yaw = np.array([r['yaw_error_deg'] for r in values])
        metrics = {f'recall_{t:g}m':float((xy<=t).mean()) for t in (.1,.5,1)}
        metrics.update(recall_1m_30deg=float(((xy<=1)&(yaw<=30)).mean()),
            median_xy_m=float(np.median(xy)),median_yaw_deg=float(np.median(yaw)))
        for outcome in ('repair','harm','preserve_success','preserve_failure'):
            metrics[outcome] = sum(v['vs_vision_1m']==outcome for v in values)
        result[method] = metrics
    return result
