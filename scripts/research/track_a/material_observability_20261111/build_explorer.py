"""Offline explorer: display aggregates from actual context0 raw RIRs, no interpolation."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=json.loads((a.root/'results.json').read_text());rows=json.loads((a.root/'features.json').read_text())['rows'];manifest=json.loads((a.root/'controlled/manifest.json').read_text());byname={v['name']:v for v in manifest};prediction={v['name']:v for v in result['predictions']};selection=result['representative'];chosen=[selection['default']]+selection['exceptions'];records={};checks=[]
 def tidy(x):return [float(f'{float(v):.6g}') for v in x]
 for row in rows:
  if row['pose']['id'] not in chosen or row['context']!=0 or row['mode']!='both':continue
  rec=byname[row['name']];path=a.root/'controlled'/row['name']/'rir.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['rir_sha256'];arrays=[]
  with np.load(path) as z:
   for c in range(6):
    x=z[f'ch{c}'].astype(float);power=x*x;envelope=lfilter(np.ones(6)/6,[1],power);views={}
    for name,length in [('early',3840),('full',38400)]:
     wave=np.zeros(length);env=np.zeros(length);n=min(len(x),length);wave[:n]=x[:n];env[:n]=envelope[:n];bw=wave.reshape(480,-1);be=env.reshape(480,-1);views[name]=dict(lo=tidy(bw.min(1)),hi=tidy(bw.max(1)),env=tidy(be.max(1)),peak=float(abs(wave[48:] if name=='early' else wave).max()),emax=float((env[48:] if name=='early' else env).max()))
    edc=np.cumsum(power[::-1])[::-1];db=10*np.log10(np.maximum(edc/max(edc[0],1e-30),1e-12));ind=np.minimum(np.arange(480)*80,len(x)-1);ed=tidy(db[ind]);spectrum=abs(np.fft.rfft(x[48:3840],n=4096))**2/4096;ids=np.unique(np.round(np.geomspace(2,1706,128)).astype(int));arrays.append(dict(views=views,edc_db=ed,duration_ms=len(x)/48,spectrum=tidy(spectrum[ids])))
  key='|'.join([row['pose']['id'],f"{row['absorption']:g}",f"{row['scattering']:g}",row['profile'],row['state']]);records[key]=dict(name=row['name'],pose=row['pose'],geometry=rec['geometry'],absorption=row['absorption'],scattering=row['scattering'],profile=row['profile'],state=row['state'],material=rec['material']['materials'][0],features=row['features'],scores=prediction[row['name']]['scores'],decisions=prediction[row['name']]['predictions'],arrays=arrays,raw_sha256=rec['rir_sha256']);checks.append(dict(key=key,name=row['name'],raw_sha256=rec['rir_sha256']))
 assert len(records)==108*len(chosen)
 packet=dict(records=records,selection=selection,poses=chosen,effects=result['effects'],models=result['models'],frequency_hz=(ids*48000/4096).tolist(),bands_hz=json.loads((a.root/'features.json').read_text())['band_edges_hz'],scope='Six designed controlled poses, default representative plus exceptions; HSSD transfer is a separate stage-only audit')
 raw=json.dumps(packet,separators=(',',':'),allow_nan=False).encode();encoded=base64.b64encode(gzip.compress(raw,compresslevel=9)).decode();template=Path(__file__).with_name('explorer_template.html').read_text();a.out.write_text(template.replace('__PAYLOAD__',encoded));(a.root/'explorer_manifest.json').write_text(json.dumps(dict(records=len(records),poses=chosen,source_records=checks,uncompressed_bytes=len(raw),html_bytes=a.out.stat().st_size,payload_sha256=hashlib.sha256(raw).hexdigest(),html_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),display='480 bin extrema/envelope, six significant digits; EDC and128 frequency samples; metrics from full raw arrays; context0 deterministic choice; no generated/interpolated RIR'),indent=2));print(len(records),a.out.stat().st_size)
if __name__=='__main__':main()
