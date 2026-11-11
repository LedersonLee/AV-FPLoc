"""Download pinned official ESC50 data, preserve license, make category-disjoint noise bank."""
import csv,hashlib,io,json,urllib.request,zipfile,argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
COMMIT='33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);url=f'https://codeload.github.com/karolpiczak/ESC-50/zip/{COMMIT}';archive=a.out/'source.zip';urllib.request.urlretrieve(url,archive);z=zipfile.ZipFile(archive);prefix=z.namelist()[0].split('/')[0]+'/'
 for name in ['LICENSE','README.md','meta/esc50.csv']:(a.out/Path(name).name).write_bytes(z.read(prefix+name))
 rows=list(csv.DictReader(io.StringIO((a.out/'esc50.csv').read_text())));assert len(rows)==2000;categories=sorted({r['category'] for r in rows});fixed={'laughing','dog','rain','siren','clock_alarm'};assert fixed<=set(categories)
 held=fixed|set(sorted(set(categories)-fixed,key=lambda c:hashlib.sha256(c.encode()).hexdigest())[:5]);assert len(held)==10;train=[r for r in rows if r['category'] not in held];dev=[r for r in rows if r['category'] in held];receipts=[]
 for part,rr in [('train',train),('held_noise',dev)]:
  out=np.lib.format.open_memmap(a.out/f'{part}.npy',mode='w+',dtype='float32',shape=(len(rr),240000))
  for i,r in enumerate(rr):
   data=z.read(prefix+'audio/'+r['filename']);sr,x=wavfile.read(io.BytesIO(data));assert sr==44100 and x.shape==(220500,) and x.dtype==np.int16;y=resample_poly(x.astype(np.float64)/32768,160,147).astype('float32');assert y.shape==(240000,) and np.isfinite(y).all();out[i]=y;receipts.append(dict(split=part,index=i,file=r['filename'],category=r['category'],source_sha256=hashlib.sha256(data).hexdigest()))
  out.flush();del out
 result=dict(repository='https://github.com/karolpiczak/ESC-50',commit=COMMIT,url=url,zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),license='CC BY-NC3.0 dataset; individual attribution preserved in LICENSE; local noncommercial research only',train_categories=sorted(set(categories)-held),held_noise_categories=sorted(held),split_rule='5 named paper example categories plus5 SHA256-ordered categories; independent implementation assumption, not author split',noise_spatial_assumption='one mono ESC clip shared across6 microphones; Gaussian noise independent across channels',records=receipts)
 (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',len(train),len(dev),flush=True)
if __name__=='__main__':main()
