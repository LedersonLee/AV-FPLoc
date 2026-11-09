"""Verify fixed-only evaluation against the previously recorded fixed arms."""
import argparse,json,hashlib
from pathlib import Path

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 setup=read(a.run/'protocol.json');old=read(a.reference/'protocol.json')
 assert setup['fixed_only'] and setup['visual_weights']==[0.,.85,1.] and setup['temperatures']==[1.]
 assert read(a.run/'selection.json')=={} and read(a.run/'summary.json')['selection']=={}
 for k in ['checkpoint_hashes','cache_manifest_sha256','scene_ids','seeds']:assert setup[k]==old[k],k
 methods={'visual','acoustic','fixed','fixed_audio_shuffle'}
 def key(r):return tuple(r[k] for k in ('scene','seed','query','method'))
 new=read(a.run/'predictions.json');ref=[r for r in read(a.reference/'predictions.json') if r['method'] in methods]
 assert {r['method'] for r in new}==methods
 nr={key(r):r for r in new};rr={key(r):r for r in ref}
 assert len(nr)==len(new) and len(rr)==len(ref) and nr.keys()==rr.keys()
 assert nr==rr,'Fixed-arm predictions or metrics changed'
 expected=len(setup['scene_ids'])*len(setup['seeds'])*64*2
 assert read(a.run/'summary.json')['unimodal_replay_checks']==expected
 result=dict(rows_exact=len(nr),unimodal_replay_checks=expected,selection_empty=True,methods=sorted(methods),reference_summary_sha256=hashlib.sha256((a.reference/'summary.json').read_bytes()).hexdigest(),run_summary_sha256=hashlib.sha256((a.run/'summary.json').read_bytes()).hexdigest())
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
