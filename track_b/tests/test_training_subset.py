import pytest
from track_b.data.training_subset import restrict_training_scenes

def test_training_subset_keeps_frozen_order_and_full_val():
    s={'train':['b','a','c'],'val':['v','w']}
    selected=restrict_training_scenes(s,2)
    assert selected=={'train':['b','a'],'val':['v','w']} and len(s['train'])==3
    for value in [0,4,-1,True,1.5]:
        with pytest.raises(ValueError):restrict_training_scenes(s,value)
    with pytest.raises(ValueError):restrict_training_scenes({'train':['a'],'val':['a']},1)
