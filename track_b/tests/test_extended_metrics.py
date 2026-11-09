import math
import pytest
import torch
from track_b.eval.extended_metrics import extended_pose_statistics,reliability_statistics

def test_wrap_precision_topk_and_prediction_centered_confidence():
 poses=torch.tensor([[0.,0.,math.pi-.05],[.4,0.,-math.pi+.05],[2.,0.,0.]])
 score=torch.tensor([3.,2.,1.]);gt=torch.tensor([0.,0.,-math.pi+.05])
 out=extended_pose_statistics(score,poses,gt,diverse=False)
 assert out['joint']==out['raw1']==out['raw5']==out['raw10']==1
 assert out['strict_02_10']==out['strict_05_15']==1
 assert out['yaw_error_deg']==pytest.approx(math.degrees(.1),abs=1e-4)
 assert out['joint_region_confidence']==pytest.approx(float(score.softmax(0)[:2].sum()))
 assert out['true_grid_rank']==1 and out['true_grid_score_ties']==1
 changed=extended_pose_statistics(score,poses,torch.tensor([99.,0.,0.]),diverse=False)
 assert changed['joint']==0 and changed['joint_region_confidence']==out['joint_region_confidence']

def test_reliability_edges_and_selective_risk():
 rows=[dict(joint_region_confidence=c,joint=y,translation_error_m=e,yaw_error_deg=e*10) for c,y,e in [(1.,1.,0.),(.8,1.,.1),(.2,0.,2.),(0.,0.,3.)]]
 result=reliability_statistics(rows,bins=2)
 assert result['ece']==pytest.approx(.1)
 assert result['bins'][1]['count']==2 and result['selective_risk'][2]['risk']==0
 assert result['selective_risk'][-1]['risk']==.5
 assert result['translation_median_m']==pytest.approx(1.05)

def test_single_candidate_and_tied_rank():
 out=extended_pose_statistics(torch.tensor([0.]),torch.zeros(1,3),torch.zeros(3),diverse=False)
 assert out['normalized_entropy']==0 and out['joint_region_confidence']==1
 out=extended_pose_statistics(torch.zeros(2),torch.tensor([[0.,0.,0.],[2.,0.,0.]]),torch.tensor([2.,0.,0.]),diverse=False)
 assert out['true_grid_rank']==1 and out['true_grid_score_ties']==2
