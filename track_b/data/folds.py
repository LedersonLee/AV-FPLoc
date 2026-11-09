"""Scene-family separated OOF partitions restricted to original training scenes."""
def validate_training_fold(base_scenes,fold,source_sha256):
    if fold.get('role')!='out_of_fold_train_scenes' or fold.get('source_cache_protocol_sha256')!=source_sha256:
        raise ValueError('Fold role/source differs from frozen input cohort')
    selected={part:fold.get(part,[]) for part in ['train','val']}
    if any(not isinstance(v,list) or not v or len(v)!=len(set(v)) for v in selected.values()):raise ValueError('Empty or duplicated scene list')
    train=set(selected['train']);held=set(selected['val']);pool=set(base_scenes['train'])
    if train&held or train|held!=pool:raise ValueError('OOF fold must partition original Train only')
    if {s.split('_')[0] for s in train}&{s.split('_')[0] for s in held}:raise ValueError('Source-family leakage across fold')
    return selected
