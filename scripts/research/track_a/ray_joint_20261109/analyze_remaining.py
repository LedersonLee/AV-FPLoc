"""Paired Transfer, common-output-subset, optimization and available fullbank analyses."""
from pathlib import Path
import json,contextlib,io
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
 from analyze_extensions import summarize,difference
P=Path(__file__).resolve().parent;W=P.parent.parent
read=lambda name:json.loads((P/name).read_text())
primary=read('candidate_evaluation.json');ext=read('extension_analysis.json');out={}
tr=ext['transfer'];out['transfer_contrasts']={}
for name,left,right in [('chirp-raw','acoustic/chirp','acoustic/raw'),('attention-concat','joint/attention','joint/concat'),('attention-disabled','joint/attention','joint/disabled'),('attention-acoustic','joint/attention','joint/acoustic')]:
 out['transfer_contrasts'][name]=difference(tr[left+'/matched/uniform/visual_geometry'],tr[right+'/matched/uniform/visual_geometry'])
rr=[r for r in primary['records'] if r['control']=='matched' and r['method']=='visual_geometry'];ss=summarize(rr,['group','kind','trained_k','eval_k']);out['common_subset_contrasts']={}
for kind in ['ecn','embedding','raw','chirp']:
 for k in [4,8,12]:
  out['common_subset_contrasts'][f'{kind}/trained36subset{k}-native{k}']=difference(ss[f'acoustic/{kind}/36/{k}'],ss[f'acoustic/{kind}/{k}/{k}'])
  out['common_subset_contrasts'][f'{kind}/all36-subset{k}']=difference(ss[f'acoustic/{kind}/36/36'],ss[f'acoustic/{kind}/36/{k}'])
follow=read('candidate_followup_evaluation.json');summary=summarize([r for r in follow['records'] if r['control']=='matched' and r['method']=='visual_geometry' and r['eval_k']==36],['kind']);out['optimization_summary']=summary;out['optimization_contrasts']={}
for a,b in [('attention','concat'),('attention','disabled')]:out['optimization_contrasts'][a+'160-'+b+'160']=difference(summary[a],summary[b])
for kind in ['concat','attention','disabled']:out['optimization_contrasts'][kind+'160-'+kind+'40']=difference(summary[kind],ss[f'joint/{kind}/36/36'])
old=read('followup_analysis.json')
for kind in summary:
 for metric in ['joint','fine','pair_accuracy']:assert summary[kind][metric]['mean']==old[kind][metric]['mean']
prior=json.loads((W/'experiments/directional_structure_20261109/discrimination_results.json').read_text());baseline=[r for r in prior['records'] if r['arm'] in ['visual','cosine','visual_cosine','B2','GEOQ']];out['development_baselines']=summarize(baseline,['kind','arm'])
for name,keys in [('fullbank_evaluation.json',['group','kind','trained_k','eval_k','method']),('conditioned_fullbank_results.json',['group','kind','control','method','fusion'])]:
 if (P/name).exists():out[name]=summarize(read(name)['records'],keys,['joint','fine','R1','top5','top20','repair','harm'])
if 'fullbank_evaluation.json' in out:
 fb=out['fullbank_evaluation.json'];out['primary_fullbank_contrasts']={}
 for left,right in [('acoustic/chirp','acoustic/raw'),('joint/attention','joint/concat'),('joint/attention','joint/disabled')]:
  out['primary_fullbank_contrasts'][left+' minus '+right]=difference(fb[left+'/36/36/visual_geometry'],fb[right+'/36/36/visual_geometry'])
 priorfb=json.loads((W/'experiments/directional_structure_20261109/fullbank_results.json').read_text())['records'];bases=summarize([r for r in priorfb if r['arm'] in ['visual','cosine','visual_cosine']],['arm'],['joint','fine','R1','top5','top20']);out['fullbank_baselines']=bases
 for model in ['acoustic/raw','acoustic/chirp','acoustic/embedding','joint/attention']:
  out['primary_fullbank_contrasts'][model+' minus visual']=difference(fb[model+'/36/36/visual_geometry'],bases['visual'])
if (P/'direction_permutation_results.json').exists():
 perm=read('direction_permutation_results.json');out['direction_permutation']=summarize(perm['records'],['group','kind','method','fusion']);out['direction_permutation_contrasts']={}
 for key,v in out['direction_permutation'].items():
  group,kind,method,fusion=key.split('/');matched=ext['conditioned'][f'{group}/{kind}/matched/{method}/{fusion}'];out['direction_permutation_contrasts'][key+' matched-minus-permuted']=difference(matched,v)
(P/'remaining_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['transfer_contrasts'],indent=2))
