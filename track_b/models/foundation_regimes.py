"""Explicit precursor-encoder tuning boundaries; new interaction heads always learn."""
def set_foundation_regime(model, regime):
    if regime not in ('frozen','partial','joint'):
        raise ValueError(regime)
    for p in model.parameters():
        p.requires_grad_(True)
    if regime != 'joint':
        for module in (model.visual, model.acoustic, model.floorplan, model.fp_head,
                       model.acoustic_floorplan, model.acoustic_fp_head):
            for p in module.parameters():
                p.requires_grad_(False)
    if regime == 'partial':
        for module in (model.visual.head[-2], model.acoustic.head[-2],
                       model.fp_head[-1], model.acoustic_fp_head[-1]):
            for p in module.parameters():
                p.requires_grad_(True)
    # Both score scales and new projections/retrieval/gate/joint heads remain free.
    return {name: p.numel() for name,p in model.named_parameters() if p.requires_grad}
