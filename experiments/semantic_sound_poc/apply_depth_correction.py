"""Create corrected inference run without overwriting the initial failed audit."""
import argparse,json,shutil,hashlib,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();old=a.out;new=old/'validated';assert not new.exists(),'Use a fresh output directory';new.mkdir();d=json.loads((old/'source_depth_clearance.json').read_text());support=json.loads((old/'source_support.json').read_text());invalid={}
for row in d['records']:
 if not row['pass_clearance']:invalid.setdefault((row['room'],row['group']),set()).add(row['index'])
for room in support:
 for group in ['generation','integration']:room[group]=[p for i,p in enumerate(room[group]) if i not in invalid.get((room['room'],group),set())];assert room[group]
for name in ['config.json','inference_candidates.npz','evaluation_truth.npz','input_manifest.json','images.json','alignment.json','environment.json','geometry_validation.json','source_depth_clearance.json','acoustic_pairs.json','acoustic_render.json','direct_probe.json','collision_backend_check.json','collision_backend_physics_check.json','receiver_radius_audit.txt','protocol.md']:
 shutil.copy2(old/name,new/name)
for name in ['rirs','direct_probe']:(new/name).symlink_to(old/name,target_is_directory=True)
(new/'source_support.json').write_text(json.dumps(support,indent=2)+'\n')
code=Path(__file__).parent;sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();correction=dict(reason='cast_ray backend false-negative floor control; replace by actual six-cardinal depth clearance at height1.2m',invalid_source_points=d['invalid'],original_source_points=d['count'],retained_source_points=sum(len(r[g]) for r in support for g in ['generation','integration']),initial_metrics_path=str(old/'metrics.json'),initial_metrics_sha256=hashlib.sha256((old/'metrics.json').read_bytes()).hexdigest(),initial_delta_pp=json.loads((old/'metrics.json').read_text())['metrics'][3]['delta_pp'],execution_sha=sha,threshold_m=.15,threshold_preexisting=True,native_audio_scope='same preserved source positions and RIR; selected RIR source positions independently confirmed in depth-valid pool',raw_parent=str(old))
# Ensure native RIR sources remain valid after correction.
for pair in json.loads((old/'acoustic_pairs.json').read_text()):assert pair['source'] in support[pair['room']]['generation']
(new/'correction.json').write_text(json.dumps(correction,indent=2)+'\n');(old/'CORRECTION.md').write_text('# Initial result superseded\n\nNative cast_ray returned no hits even for floor controls. Six-direction rendered depth found16 of1108 support points within the predeclared15cm boundary. Corrected source support and all Oracle baselines are in validated/. Initial +4.84pp is retained as exploratory pre-correction evidence, not the final result. Raw acoustic results retained; all three selected acoustic sources pass final depth clearance. No performance-based tuning.\n')
print(new)
