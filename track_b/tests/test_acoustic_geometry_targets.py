import numpy as np
from track_b.acoustic_geometry.targets import intersect_segments


def test_room_orientation_translation_and_opening():
    wall=np.array([[[0,0],[4,0]],[[4,0],[4,6]],[[4,6],[0,6]],[[0,6],[0,0]]])
    pose=np.array([[1,2,0],[1,2,np.pi/2]])
    d,m=intersect_segments(pose,wall,directions=4)
    np.testing.assert_allclose(d,[[3,4,1,2],[4,1,2,3]],atol=1e-6)
    assert m.all()
    shifted=pose.copy();shifted[:,:2]+=[12,-7]
    np.testing.assert_allclose(intersect_segments(shifted,wall+[12,-7],4)[0],d)
    d,m=intersect_segments(pose[:1],wall[[0,2,3]],4)
    assert not m[0,0] and d[0,0]==20
    d,m=intersect_segments(pose[:1],wall,4,max_range=2.5)
    assert m.tolist()==[[False,False,True,True]]
