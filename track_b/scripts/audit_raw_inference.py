"""Raw RGB/body-RIR through upstream encoder and full-field inference, six probes."""
import argparse,json,sys,time
from pathlib import Path
import cv2
import torch
from track_b.models.raw_observations import audio_features,visual_features
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.scripts.train_structure import read,sha

def main():
 p=argparse.ArgumentParser()
 for k in ['packets','cache','features','source','upstream_checkpoint','checkpoint','out']:
  p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 assert sha(a.upstream_checkpoint)=='6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d'
 sys.path.insert(0,str(a.source));from depth_anything_v2.dpt import DepthAnythingV2
 upstream=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024]);upstream.load_state_dict(torch.load(a.upstream_checkpoint,weights_only=True,map_location='cpu'));upstream.cuda().eval().requires_grad_(False)
 model=FoundationInteractionMatcher('attention').cuda().eval();model.load_state_dict(torch.load(a.checkpoint,weights_only=True))
 packets=read(a.packets/'manifest.json');assert len(packets)==6
 features_manifest={x['scene']:x['sha256'] for x in read(a.features/'manifest.json')}
 cache_manifest={x['scene']:x['cache_sha256'] for x in read(a.cache/'manifest.json')}
 events=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):events.append(str(args[0]))
 sys.addaudithook(hook);rows=[]
 import numpy as np
 with torch.no_grad():
  for packet in packets:
   sid=packet['scene'];image=a.packets/packet['rgb'];wave=a.packets/packet['body_rir'];assert sha(image)==packet['rgb_sha256'] and sha(wave)==packet['body_rir_sha256']
   sfile=a.cache/f'{sid}.pt';fpath=a.features/f'{sid}.pt';assert sha(sfile)==cache_manifest[sid] and sha(fpath)==features_manifest[sid]
   s=torch.load(sfile,weights_only=True,map_location='cpu');features=torch.load(fpath,weights_only=True,map_location='cpu');q=s['query_ids'].index(packet['query']);assert features['query_ids']==s['query_ids']
   # Only map inputs are passed; no GT, target, scene or query identifier.
   cached=model.prepare_map(s['floorplan'].cuda(),{'origin':s['origin'].cuda(),'spacing':s['spacing'],'hfov_deg':s['hfov_deg']},s['candidates'].cuda())
   rgb=cv2.imread(str(image));body=np.load(wave);assert rgb is not None
   events.clear();torch.cuda.synchronize();start=time.perf_counter()
   vf=visual_features(upstream,rgb);af=audio_features(body,packet['sample_rate_hz']).cuda()
   field=model.forward_cached(vf.float(),af.float(),cached)
   top=field.topk(20,dim=-1).indices;pose=s['candidates'][int(top[0,0]),:3]
   torch.cuda.synchronize();seconds=time.perf_counter()-start;forward_opens=list(events)
   torch.testing.assert_close(vf.cpu(),features['visual_features'][q:q+1],atol=0,rtol=0)
   torch.testing.assert_close(af.cpu(),s['spectrum'][q:q+1],atol=0,rtol=0)
   reference=model.forward_cached(features['visual_features'][q:q+1].cuda().float(),s['spectrum'][q:q+1].cuda().float(),cached)
   torch.testing.assert_close(field,reference,atol=0,rtol=0)
   unexpected=[path for path in forward_opens if not path.startswith('/usr/local/lib/python3.10/site-packages/torch/')];assert not unexpected,unexpected
   rows.append(dict(scene=sid,query=packet['query'],candidates=field.shape[-1],raw_feature_bitwise_equal=True,spectrum_bitwise_equal=True,full_field_bitwise_equal=True,raw_top20_equal=bool(torch.equal(top,reference.topk(20,dim=-1).indices)),decoded_observation_to_map_seconds=seconds,python_forward_open_paths=forward_opens,output_pose_shape=list(pose.shape)))
   print('DONE',sid,flush=True)
 result=dict(checkpoint_sha256=sha(a.checkpoint),upstream_checkpoint_sha256=sha(a.upstream_checkpoint),packet_manifest_sha256=sha(a.packets/'manifest.json'),rows=rows,scope='one fixed first-query per six reused Val scenes, attention seed42; raw decoded RGB/body-frame RIR through frozen upstream, trainable adapters and full bank/MAP. Map cache, model load and image/wave disk decode excluded from timing. Concurrent authorized training; timings are observations not isolated cost benchmarks. Python audit hooks do not trace native library I/O.',dataset_boundary='body-frame waveform materialized offline using schedule orientation; inference receives no world pose. Calibration feasibility for new physical recordings not established.',test_accessed=False)
 (a.out/'audit.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
