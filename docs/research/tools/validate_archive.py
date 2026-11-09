#!/usr/bin/env python3
"""Validate the committed research archive without running any experiment."""
from pathlib import Path
import argparse,hashlib,json,re
from urllib.parse import unquote
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path);args=ap.parse_args();errors=[];copies=0;sources=0;tables=0
 for f in P.rglob('*.json'):
  try:json.loads(f.read_text())
  except Exception as e:errors.append(f'{f}: {e}')
 for f in P.rglob('*.md'):
  body=re.sub(r'```.*?```','',f.read_text(),flags=re.S)
  for url in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)',body):
   url=url.strip('<>')
   if url.startswith(('https://','http://','mailto:','#')):continue
   target=(f.parent/unquote(url.split('#')[0])).resolve()
   if not target.exists():errors.append(f'Broken link {f.relative_to(P)} -> {url}')
 for m in P.glob('track-a/experiments/*/artifacts.json'):
  r=json.loads(m.read_text())
  for e in r['source_files']:
   if e['git_copy']:
    dest=m.parent/e['git_copy'];copies+=1
    if not dest.exists() or sha(dest)!=e['git_copy_sha256']:errors.append(f'Git copy hash mismatch: {dest}')
   if args.source_root:
    src=args.source_root/'experiments'/r['study_id']/e['path'];sources+=1
    if not src.exists() or sha(src)!=e['sha256']:errors.append(f'Source hash mismatch: {src}')
    if src.suffix=='.json' and e['git_copy']:
     original=json.loads(src.read_text());copied=json.loads((m.parent/e['git_copy']).read_text());export=copied.pop('_archive_export')
     if r['study_id'].startswith('pairwise_acoustic') and src.name=='paired_statistics.json':original['records']=[x for x in original['records'] if x['group']=='all']
     if r['study_id'].startswith('structural_evidence') and src.name=='final_evaluation.json':original={k:v for k,v in original.items() if k in ['metrics_order','selected','contrasts']}
     if src.name=='oof_results.json':original['summary']=[x for x in original['summary'] if x['group']=='all']
     text=json.dumps(original,ensure_ascii=False).replace(str(args.source_root.resolve()),'${WORKSPACE}').replace('/workspace','${WORKSPACE}').replace('/file2/jeongeon/AV-FPLoc/datasets/HSSD','${HSSD_ROOT}').replace('/ssd1/AV-FPLoc/datasets/HSSD','${HSSD_SSD_ROOT}')
     if json.loads(text)!=copied:errors.append(f'JSON value/projection mismatch: {src}')
    if src.suffix=='.html' and e['git_copy']=='results.md':
     from bs4 import BeautifulSoup
     soup=BeautifulSoup(src.read_text(),'html.parser');md=(m.parent/'results.md').read_text()
     for table in soup.find_all('table'):
      tables+=1
      for tr in table.find_all('tr'):
       cells=[re.sub(r'\s+',' ',c.get_text(' ',strip=True)).replace('|','\\|') for c in tr.find_all(['th','td'],recursive=False)]
       if cells and '| '+' | '.join(cells)+' |' not in md:errors.append(f'Table row mismatch: {src.name}: {cells}')
 assert len(list(P.glob('track-a/experiments/*/artifacts.json')))==13
 pilot_copies=0
 for manifest in P.glob('track-*/experiments/*/provenance.json'):
  for entry in json.loads(manifest.read_text())['files']:
   target=manifest.parent/entry['git_copy'];pilot_copies+=1
   if not target.exists() or sha(target)!=entry['sha256']:errors.append(f'Pilot copy hash mismatch: {target}')
 if errors:
  print('\n'.join(errors));raise SystemExit(1)
 print(f'PASS: 13 studies; {copies} Git copy hashes; {sources} source hashes; {tables} original HTML tables; {pilot_copies} imported artifact hashes; JSON and local Markdown links valid.')
if __name__=='__main__':main()
