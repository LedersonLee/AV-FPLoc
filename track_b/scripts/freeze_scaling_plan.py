"""Freeze a2x2 scene-count/training-budget control before scaled model outcomes."""
import argparse,json,hashlib
from pathlib import Path
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['cache','pilot','validation','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();data=read(a.cache/'protocol.json');pilot=read(a.pilot/'protocol.json');validation=read(a.validation/'validation.json')
 assert validation['scenes']==131 and validation['queries']==8384 and validation['pilot_tensor_and_input_manifests_exact']
 assert data['scenes']['train'][:16]==pilot['scenes']['train'] and data['scenes']['val'][:6]==pilot['scenes']['val']
 assert len(data['scenes']['train'])==112 and len(data['scenes']['val'])==19
 a.out.mkdir(parents=True,exist_ok=False);conditions=[]
 for count in [16,112]:
  for multiplier in [1,7]:
   aux=1500*multiplier;dense=1000*multiplier;name=f'n{count}_budget{multiplier}'
   stages=[dict(stage='visual_aux',mode='visual',aux_steps=aux,dense_steps=0,foundation_features=True,pretrained_stage=None,aux_weight=10,fp_anchor_weight=0,skip_sampling_steps=0),dict(stage='visual_joint',mode='visual',aux_steps=0,dense_steps=dense,foundation_features=True,pretrained_stage='visual_aux',aux_weight=1000,fp_anchor_weight=1000,skip_sampling_steps=aux),dict(stage='acoustic',mode='acoustic',aux_steps=aux,dense_steps=dense,foundation_features=False,pretrained_stage=None,aux_weight=10,fp_anchor_weight=0,skip_sampling_steps=0)]
   conditions.append(dict(name=name,train_scene_count=count,budget_multiplier=multiplier,train_scenes=data['scenes']['train'][:count],val_scenes=data['scenes']['val'],stages=stages,expected_aux_draws_per_train_query=8*aux/(count*64),expected_dense_draws_per_train_query=8*dense/(count*64)))
 result=dict(conditions=conditions,seeds=[42,43,44],batch=8,lr=.0003,weight_decay=.0001,scene_residency='cpu',save_initial_state=True,scheduled_stage_fits=36,final_modality_models=24,cache_protocol_sha256=sha(a.cache/'protocol.json'),cache_validation_sha256=sha(a.validation/'validation.json'),selection='scheduled final checkpoints only; no epoch/model/grid/calibration tuning from expanded Val',scorer='independent visual/acoustic dual-FP fixed .85/.15; not M1 interaction',evaluation='all19 Val plus prespecified original6/added13 strata; no independent-Test claim',primary_contrasts='112vs16 at each fixed update budget; budget7vs1 at each count;112/budget7 vs16/budget1 at matched expected draws per training query',limitation='budget factor scales both aux and dense learning, not an isolated dense-only effect; repeated diagnostics on full19Val are not used to select updates; upstream asset exposure unresolved',start_gate='require complete expanded feature verification with exact22-scene pilot overlap; failed partial outputs must be preserved',test_accessed=False)
 (a.out/'plan.json').write_text(json.dumps(result,indent=2));print('COMPLETE')
if __name__=='__main__':main()
