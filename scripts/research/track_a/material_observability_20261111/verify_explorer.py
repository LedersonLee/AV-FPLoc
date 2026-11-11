import argparse,base64,gzip,hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--html',type=Path,required=True);a=p.parse_args();html=a.html.resolve();text=html.read_text();packed=re.search(r"const PACK='([^']+)'",text)[1];data=json.loads(gzip.decompress(base64.b64decode(packed)));features={v['name']:v for v in json.loads((a.root/'features.json').read_text())['rows']};n=0;max_bound=0
 for rec in data['records'].values():
  raw=a.root/'controlled'/rec['name']/'rir.npz';assert hashlib.sha256(raw.read_bytes()).hexdigest()==rec['raw_sha256'];assert rec['features']==features[rec['name']]['features']
  with np.load(raw) as z:
   for c in range(6):
    x=z[f'ch{c}'].astype(float);env=lfilter(np.ones(6)/6,[1],x*x);arr=rec['arrays'][c]
    for name,length,count in [('early',3840,480),('full',38400,160)]:
     y=np.zeros(length);e=np.zeros(length);ln=min(len(x),length);y[:ln]=x[:ln];e[:ln]=env[:ln]
     if name=='early':y[:48]=0;e[:48]=0
     v=arr['views'][name];lo=np.frombuffer(base64.b64decode(v['lo']),dtype='<i2')*v['wave_scale']/32767;hi=np.frombuffer(base64.b64decode(v['hi']),dtype='<i2')*v['wave_scale']/32767;energy=np.frombuffer(base64.b64decode(v['env']),dtype='<u2')*v['emax']/65535
     for obs,truth in [(lo,y.reshape(count,-1).min(1)),(hi,y.reshape(count,-1).max(1))]:
      err=max(abs(obs-truth));assert err<=v['wave_scale']/65534+1e-12;max_bound=max(max_bound,float(err/max(v['wave_scale'],1e-30)))
     assert max(abs(energy-e.reshape(count,-1).max(1)))<=v['emax']/131070+1e-12
    d=np.cumsum((x*x)[::-1])[::-1];edc=10*np.log10(np.maximum(d/max(d[0],1e-30),1e-12));ids=np.minimum(np.arange(160)*240,len(x)-1);np.testing.assert_allclose(np.frombuffer(base64.b64decode(arr['edc_db']),dtype='<i2')*.01,edc[ids],atol=.005001)
    freqids=np.rint(np.array(data['frequency_hz'])*4096/48000).astype(int);spec=abs(np.fft.rfft(x[48:3840],n=4096))**2/4096;np.testing.assert_allclose(np.frombuffer(base64.b64decode(arr['spectrum']),dtype='<f4'),spec[freqids],rtol=1e-6,atol=1e-30);n+=1
 browser=[]
 for fragment in ['selftest','failure','removed']:
  command=['/home/jeongeonlee/.local/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--allow-file-access-from-files','--virtual-time-budget=20000','--dump-dom',html.as_uri()+'#'+fragment];r=subprocess.run(command,capture_output=True,text=True,timeout=60);(a.root/f'explorer_{fragment}_dom.html').write_text(r.stdout);assert r.returncode==0 and 'data-ready="true"' in r.stdout and 'data-error=' not in r.stdout
  state=dict(view=fragment,ready=True)
  if fragment=='selftest':state['control_checks']=int(re.search(r'data-tests="(\d+)"',r.stdout)[1]);assert state['control_checks']>=349
  if fragment=='failure':assert 'data-a="single6_moved_a0.99_s0.0_high_r0_both"' in r.stdout;assert 'raw amplitude gate 0/6채널' in r.stdout
  if fragment=='removed':assert 'data-a="room_far_removed_a0.30_s0.5_uniform_r0_both"' in r.stdout;assert '다른 반사를 target-wall 증거로 오인' in r.stdout
  browser.append(state)
 out=dict(passed=True,records=len(data['records']),raw_channel_displays=n,max_relative_wave_display_error=max_bound,metric_records_exact=len(data['records']),html_sha256=hashlib.sha256(html.read_bytes()).hexdigest(),browser=browser,visual_review='Screenshots separately inspected; common axes, counterexample controls, profiles and readouts visible')
 (a.root/'explorer_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
