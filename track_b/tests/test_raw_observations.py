import inspect
import numpy as np
import pytest
import torch
from track_b.models.raw_observations import audio_features,visual_features

def test_body_frame_audio_input_contract():
    x=np.zeros((6,2000),np.float32);x[:,100]=np.arange(1,7)
    f=audio_features(x)
    assert f.shape==(1,18,44,513) and torch.isfinite(f).all()
    assert not torch.equal(f,audio_features(np.roll(x,1,axis=0)))
    assert torch.isfinite(audio_features(np.zeros_like(x))).all()
    with pytest.raises(ValueError):audio_features(x,16000)
    with pytest.raises(ValueError):audio_features(x[:5])
    for fn in [audio_features,visual_features]:
        assert not any(k in inspect.signature(fn).parameters for k in ['gt','yaw','scene_id','query_id'])
