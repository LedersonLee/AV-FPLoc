"""Validate public display assets and exact candidate metadata before publishing."""
import gzip,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_text());checks=0;worst=0
load=lambda p:json.loads(gzip.decompress(p.read_bytes()))
for q in m['queries']:
 sid=q['scene'];sample=q['sample'];b=load(root/sid/'bank.json.gz');d=load(root/sid/f'{sample}.json.gz');n=len(b['xy'])*36
 packed=np.frombuffer(gzip.decompress((root/sid/f'{sample}.bin.gz').read_bytes()),dtype='<u2');assert len(packed)==n*len(d['fields'])
 for f in d['fields']:
  a=packed[f['offset']:f['offset']+f['count']].astype(float)*(f['max']-f['min'])/65535+f['min'];assert len(a)==n;assert 0<=f['top1']<n;assert a.max()-a[f['top1']]<=2*f['error_bound']+1e-10;assert f['max_error']<=f['error_bound']+1e-10;worst=max(worst,f['max_error']);checks+=1
 for seed in m['seeds']:
  s=d['seeds'][str(seed)];ids={c['id'] for c in s['candidates']};assert all(0<=k<n for k in ids);assert s['selected'] in ids
  for stage,ii in s['stages'].items():
   assert len(ii)==20;assert set(ii)<=ids
   for c in s['candidates']:assert c['ranks'][stage]==[j+1 for j,k in enumerate(ii) if k==c['id']]
  for c in s['candidates']:
   assert c['pose']==[*b['xy'][c['id']//36],c['id']%36*10]
   for sl in c['support']:assert s['stages']['support'][sl['slot']]==c['id'];assert sl['refined_id']==c['id']
  slots=sorted([sl for c in s['candidates'] for sl in c['support']],key=lambda x:x['slot']);assert len(slots)==20;gain=np.array([x['geoq_gain'] for x in slots]);assert gain[0]==0;ch=int(gain.argmax());ch=ch if gain[ch]>s['geoq_threshold'] else 0;assert ch==s['geoq_choice'];assert s['stages']['support'][ch]==s['selected'];checks+=1
 im=Image.open(root/sid/f'{sample}.webp');assert im.width<=384 and im.height<=256;assert not im.getexif()
# Review every published textual payload, including compressed metadata, for local paths / secret markers.
for p in root.rglob('*'):
 if not p.is_file():continue
 assert p.suffix not in ['.pt','.npy','.npz','.wav','.joblib','.pkl'],p
 if p.name.endswith('.json.gz'):text=gzip.decompress(p.read_bytes()).decode()
 elif p.suffix in ['.json','.md','.js','.html']:text=p.read_text()
 else:continue
 for bad in ['/home/','/file2/','/ssd1/','/workspace/','ghp_','github_pat_','-----BEGIN PRIVATE KEY']:assert bad not in text,(p,bad)
result=dict(status='pass',scenes=len(m['scenes']),queries=len(m['queries']),seeds=m['seeds'],checks=checks,max_display_error=worst,total_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),files=sum(p.is_file() for p in root.rglob('*')),training=False,rendering=False,test_accessed=False)
print(json.dumps(result,indent=2))
