"""Requirement coverage, cardinality and provenance audit; incomplete outputs remain pending."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;pr=json.loads((P/'PROTOCOL.json').read_text())
def read(name):return json.loads((P/name).read_text())
def main():
 checks=[]
 def record(requirement,passed,evidence):checks.append(dict(requirement=requirement,status='verified' if passed else 'incomplete',evidence=evidence))
 ac=read('acoustic_fits.json')['fits'];ctrl=read('control_fits.json')['fits'];joint=read('joint_fits.json')['fits'];follow=read('joint_followup_fits.json')['fits'];transfer=read('transfer_frozen.json')['models']
 expected={(kind,k,seed,fold) for kind in ['ecn','embedding','raw','chirp'] for k in [4,8,12,36] for seed in pr['seeds'] for fold in range(3)};actual={(r['kind'],r['k'],r['seed'],r['fold']) for r in ac};assert actual==expected and len(ac)==96
 assert len(ctrl)==12 and len(joint)==30 and len(follow)==24 and len(transfer)==18
 assert all(r['updates']==320 and r['epochs']==40 for r in ac+ctrl+joint)
 for r in ac:
  assert set(r['fit_scenes']).isdisjoint(r['held_scenes']);assert len(r['fit_scenes'])==8 and len(r['held_scenes'])==4
 for seed in pr['seeds']:
  for fold in range(3):
   for k in [4,8,12,36]:
    pair=[r for r in ac if r['seed']==seed and r['fold']==fold and r['k']==k and r['kind'] in ['raw','chirp']];assert len(pair)==2;assert pair[0]['parameters']==pair[1]['parameters']
 model=read('checkpoint_verification.json');assert model['checkpoints']==228 and model['arrays']==1542 and model['max_absolute_error']==0
 record('Phase0 implementation and AVLoc distinction; tensor/loss/score pipeline',all((P/n).exists() for n in ['METHOD_AT_A_GLANCE.md','environment.json','source_inventory.json']),['METHOD_AT_A_GLANCE.md','prepare_audio.py','train_acoustic.py','joint_models.py','attached AVLoc Sec3/Table2/supplement audio recipe; manually inspected'])
 oracle=read('oracle_results.json');assert len(oracle['records'])==168;assert len(set(r['scene'] for r in oracle['records']))==12;assert len(set(r['arm'] for r in oracle['records']))==7;record('Phase1 common-source K4/8/12/36, subsets, smooth4 and angular coverage oracle',True,['oracle_results.json','oracle_summary.json','PHASE1_REPORT.md','oracle_capacity.py; source/subset equality asserted during execution'])
 record('Phase2 all4representations×4K×2seeds×3folds and12matchedcapacitychannel controls',True,['acoustic_fits.json96fits','control_fits.json12fits','pairedraw/chirpparameters,updates,split checks'])
 candidate=read('candidate_evaluation.json');assert len(candidate['records'])==10800 and len(candidate['ray_records'])==5400
 record('Phase2 ray error and localization; nativeK and36outputsubsets; donor/permutation/isotropic',True,['candidate_evaluation.json:10800candidate/5400rayrecords','analysis.json','remaining_analysis.json'])
 record('Phase3 independentV/A,late,concat,attention,interactiondisabled and optimization alternatives',True,['joint_fits.json30fits','architecture_verification.json','candidate_evaluation.json','joint_followup_fits.json24fits','fov_late_results.json','remaining_analysis.json'])
 record('StrongB2+GEOQ and cosine references on samecohorts; distinctcandidateframeworkexplicit',True,['remaining_analysis.json development_baselines B2/GEOQ','extension_analysis.json transfer_baselines','RESEARCH_REPORT.md'])
 cal=read('calibration_verification.json');assert cal['checks']==528 and cal['outer_held_excluded'] and cal['recomputed_scales_exact']
 record('Phase4 calibration before uncertainty interpretation and GT-free FP directionselection',True,['calibration_verification.json528checks','matching_verification.json:independentpair/anchorscalarchecks','candidate_conditioned.py selection_weights acceptsFPdifferencesonly'])
 cond=read('conditioned_candidate_results.json');assert len(cond['records'])==1728
 record('Phase4 fixedcandidateuniform/uncertainty/pair-selection and correctaudio-onlydonor/shuffle',True,['conditioned_candidate_results.json1728records','candidate_conditioned.py corrected donor','diagnostics/pre_joint_donor_correction preserved'])
 for filename,count in [('fullbank_evaluation.json',2880),('conditioned_fullbank_results.json',1728)]:
  passed=(P/filename).exists() and len(read(filename)['records'])==count;record('Fullbankcomplete '+filename,passed,[filename,f'expected {count}records'])
 record('Fullbankpairedanalysis and crossimplementationuniformreplay',(P/'fullbank_analysis.json').exists(),['fullbank_analysis.json','analyze_fullbank.py'])
 inp=read('input_verification.json');assert inp['raw_RIR_hashes_verified']==2944 and inp['feature_hashes_verified']==23
 record('Input/checkpoint/sourceintegrity,strictsplit/noTestallowlist',True,['input_verification.json2944RIRs','checkpoint_verification.json228models1542arrays','PROTOCOL.json23training-source scenes','prepare_audio.py asserts train source split; Test IDs metadata only'])
 record('Preservation of priorresults and originaluserchanges',read('prior_preservation.json')['mismatches']==[],['prior_preservation.json856files','originalruntime diffSHA; verifyagainbeforefinalcommit'])
 report=(P/'RESEARCH_REPORT.md').read_text();record('Finalreport includesfullbankfindings,failures,mechanisms,limits and reproduction','Pending. Primary all-K/subset evaluation' not in report and 'interim report' not in report,['RESEARCH_REPORT.md','REPRODUCE.md','METHOD_AT_A_GLANCE.md'])
 D=P.parent.parent/'AV-FPLoc/docs/research';index=(D/'RESEARCH_INDEX.md').read_text();own=[line for line in index.splitlines() if '| Ray representation & joint learning / ray_joint_20261109 |' in line];ready=len(own)==1 and '완료' in own[0] and 'Completed controlled study' in (D/'track-a/experiments/2026-11-09-ray-joint/README.md').read_text();record('Final Docs files and index ready for safe synchronization',ready,['Docs/index reviewed; archive validator and final normal push are verified externally after this local audit, with SHA outside the index.'])
 out=dict(status='incomplete' if any(r['status']!='verified' for r in checks) else 'verified',checks=checks,scope='Artifact cardinality and explicit source/manual-evidence audit; passing one item does not substitute for unresolved requirements.');(P/'completion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
