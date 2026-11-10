"""Raw decoded RGB/body-RIR inference replay and latency on declared inner-val probes."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from .prepare_targets import read,sha
from .features import features
from .transfer import make_model,map_input,descriptors,score
from track_b.data.geometry import canonical_rir
from track_b.models.raw_observations import visual_features,audio_features


def main():
    p=argparse.ArgumentParser()
    for k in ['root','study','scale','source','checkpoint','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--repeats',type=int,default=20)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
    fp=read(a.scale/'features_v1/protocol.json');assert sha(a.checkpoint)==fp['checkpoint_sha256']
    for file,digest in fp['source_hashes'].items():assert sha(a.source/file)==digest
    sys.path.insert(0,str(a.source));from depth_anything_v2.dpt import DepthAnythingV2
    upstream=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024])
    upstream.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));upstream.cuda().eval().requires_grad_(False)
    config=read(a.study/'transfer_v1/protocol.json');registry={(r['arm'],r['seed']):r for r in read(a.study/'transfer_v1/registry.json')}
    visual=make_model(42,'visual',config['architecture']).cuda().eval()
    vp=a.study/'transfer_v1/visual_42/model.pt';assert sha(vp)==registry[('visual',42)]['checkpoint_sha256']
    visual.load_state_dict(torch.load(vp,map_location='cpu',weights_only=False)['model'])
    scenes=[config['scenes']['val'][0],config['scenes']['val'][-1]];rows=[]
    with torch.no_grad():
      for scene in scenes:
        c=torch.load(a.scale/'cache_v1'/f'{scene}.pt',map_location='cpu',weights_only=True)
        v=torch.load(a.scale/'features_v1'/f'{scene}.pt',map_location='cpu',weights_only=True)
        geom=torch.load(a.study/'features_v1'/f'{scene}.pt',map_location='cpu',weights_only=False)
        refs=read(a.study/'targets_v3'/f'{scene}_queries.json');q=refs[0]
        assert c['query_ids']==v['query_ids']==geom['query_ids'] and c['query_ids'][0]==q['query_id']
        # This offline assembly uses orientation metadata; the timed inference receives only body RIR.
        raw_path=a.root/q['rir_path'];body=canonical_rir(np.load(raw_path),q['world_yaw_deg'],q['base_acoustic_yaw_deg']).copy()
        raw_image=a.root/q['rgb_path'];rgb=cv2.imread(str(raw_image));assert rgb is not None
        # Drop GT and all diagnostic/identity tensors before preparing inference inputs.
        whitelist={k:c[k] for k in ['floorplan','origin','spacing','candidates']};mi=map_input(whitelist,'cuda')
        vf=visual_features(upstream,rgb);old=audio_features(body);_,new=features(torch.from_numpy(body)[None]);new=new.half()
        torch.testing.assert_close(vf.cpu(),v['visual_features'][:1],atol=0,rtol=0)
        torch.testing.assert_close(old,c['spectrum'][:1],atol=0,rtol=0)
        torch.testing.assert_close(new,geom['spatial'][:1],atol=0,rtol=0)
        for arm in config['arms']:
            cp=a.study/'transfer_v1'/f'{arm}_42/model.pt';assert sha(cp)==registry[(arm,42)]['checkpoint_sha256']
            model=make_model(42,arm,config['architecture']).cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
            torch.cuda.synchronize();tick=time.perf_counter();afp=descriptors(model,mi);vfp=descriptors(visual,mi);torch.cuda.synchronize();map_ms=(time.perf_counter()-tick)*1000
            def infer(visual_observation,audio_observation):
                va=visual.visual(visual_observation.float());aa=model.acoustic(audio_observation.float())
                field=.85*score(visual,va,vfp,mi['candidates'],True)+.15*score(model,aa,afp,mi['candidates'])
                return field,field.topk(20,-1).indices
            reference,rtop=infer(v['visual_features'][:1].cuda(),c['spectrum'][:1].cuda() if arm=='legacy' else geom['spatial'][:1].cuda())
            latencies=[];torch.cuda.reset_peak_memory_stats()
            for repeat in range(a.repeats+2):
                torch.cuda.synchronize();tick=time.perf_counter()
                rv=visual_features(upstream,rgb)
                if arm=='legacy':ra=audio_features(body).cuda()
                else:_,ra=features(torch.from_numpy(body)[None]);ra=ra.half().cuda()
                field,top=infer(rv,ra)
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                torch.testing.assert_close(field,reference,atol=0,rtol=0);assert torch.equal(top,rtop)
                if repeat>=2:latencies.append(elapsed)
            active_audio=sum(p.numel() for p in model.acoustic.parameters()) if arm=='legacy' else sum(p.numel() for module in [model.acoustic.geometry.encoder,model.acoustic.readout] for p in module.parameters())
            rows.append(dict(scene=scene,query=q['query_id'],arm=arm,seed=42,candidates=len(mi['candidates']),
                raw_features_bitwise_equal=True,full_field_bitwise_equal=True,top20_bitwise_equal=True,map_setup_ms=map_ms,
                decoded_rgb_body_rir_to_full_field_top20_median_ms=float(np.median(latencies)),p95_ms=float(np.quantile(latencies,.95)),
                repeats=a.repeats,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                acoustic_active_parameters=active_audio,upstream_active_parameters=sum(p.numel() for p in upstream.pretrained.parameters()),
                two_map_descriptor_bytes=(afp.numel()+vfp.numel())*afp.element_size(),
                checkpoint_sha256=sha(cp),rgb_sha256=sha(raw_image),rir_sha256=sha(raw_path)))
            del model,reference,field
    result=dict(code_sha=a.code_sha,scope='2 predeclared inner-validation scenes, first query, seed42, all6 transfer arms; 2 warmups and20 measured repeats',
        timing='Decoded RGB and offline body-aligned waveform -> upstream RGB encoder, STFT, observation branches, full-bank fixed-fusion field and rawTop20; excludes file decode/model load/map setup, which is separately timed; shared GPU',
        inference_inputs='Only decoded RGB, body RIR, known map/candidate geometry/calibration; no GT or query ID enters inference; no candidate RIR bank',
        geometry_frontend_cost='Current helper computes both Echo and Spatial features although only Spatial is used; measured overhead is included',
        upstream_checkpoint_sha256=sha(a.checkpoint),rows=rows)
    (a.out/'runtime.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
