import pytest
from track_b.data.folds import validate_training_fold

def fold(train,val):return dict(role='out_of_fold_train_scenes',source_cache_protocol_sha256='pinned',train=train,val=val)
def test_only_original_train_and_family_separation():
 base={'train':['a_1','a_2','b_1','c_1'],'val':['v_1']}
 assert validate_training_fold(base,fold(['a_1','a_2'],['b_1','c_1']),'pinned')['val']==['b_1','c_1']
 for invalid in [fold(['a_1','b_1'],['a_2','c_1']),fold(['a_1','a_2'],['b_1','v_1']),fold(['a_1','a_2'],['b_1','b_1','c_1']),fold([],base['train'])]:
  with pytest.raises(ValueError):validate_training_fold(base,invalid,'pinned')
 with pytest.raises(ValueError):validate_training_fold(base,fold(['a_1','a_2'],['b_1','c_1']),'wrong')
