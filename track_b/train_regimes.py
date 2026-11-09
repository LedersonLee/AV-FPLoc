"""Explicit frozen/partial/joint encoder tuning, applied after shared pretraining."""

def set_regime(model,regime):
    if regime not in ('frozen','partial','joint'):raise ValueError(regime)
    for p in model.parameters():p.requires_grad_(regime=='joint')
    model.log_scale.requires_grad_(True)
    if regime=='partial':
        for module in (model.visual.head[-2],model.acoustic.head[-2],model.fp_head[-1]):
            for p in module.parameters():p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
