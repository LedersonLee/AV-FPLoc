"""Deterministic Train prefix restriction with the full Val cohort unchanged."""
def restrict_training_scenes(scenes, count):
    if isinstance(count,bool) or not isinstance(count,int) or not 0<count<=len(scenes['train']):
        raise ValueError('Training scene count must be within the frozen Train set')
    if len(set(scenes['train']))!=len(scenes['train']):raise ValueError('Duplicate Train scenes')
    if set(scenes['train'])&set(scenes['val']):raise ValueError('Train/Val overlap')
    return {**scenes,'train':list(scenes['train'][:count]),'val':list(scenes['val'])}
