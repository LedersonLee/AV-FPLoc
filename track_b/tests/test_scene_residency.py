import torch
from track_b.data.scene_residency import SceneSequence

def test_scene_sequence_preserves_values_order_and_source_mapping():
    scenes=[dict(scene=str(i),rgb=torch.full((1,3,2,2),i),valid=torch.ones(2,2,dtype=torch.bool),query_ids=[i]) for i in range(3)]
    sequence=SceneSequence(scenes,device='cpu')
    assert len(sequence)==3 and [s['scene'] for s in sequence]==['0','1','2']
    assert torch.equal(sequence[2]['rgb'],scenes[2]['rgb'])
    assert sequence[0]['valid'] is scenes[0]['valid']
    sequence[0]['extra']='temporary'
    assert 'extra' not in scenes[0]
