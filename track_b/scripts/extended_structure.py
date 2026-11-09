"""Replay normal predictions, acoustic physical controls, and support-based extraction."""
import argparse,hashlib,io,json,math
from pathlib import Path
import numpy as np
import torch
from track_b.models.structure import StructureMatcher,rir_spectrum
from track_b.data.geometry import canonical_rir
from track_b.eval.metrics import pose_statistics
from track_b.eval.hypotheses import support_nms
from track_b.scripts.train_structure import scene_to_gpu,full_descriptors,scores


def audio_variants(root,cache,scene):
    sid=scene['scene'];schedule=json.loads((root/f'RIR/manifests/positive_pairs_v1/schedules/val/{sid}.json').read_text())
    queries={q['sample_index']:q for q in schedule['queries']};inputs=json.loads((cache/f'{sid}_inputs.json').read_text());variants={k:[] for k in ('direct_only','direct_masked','late_only')}
    for i,record in enumerate(inputs):
        q=queries[record['query_id']];path=root/q['observed_rir_path'];assert q['observed_rir_path'].startswith('RIR/Observed/')
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==record['rir_sha256']
        r=canonical_rir(np.load(io.BytesIO(raw)),q['world_yaw_deg'],q['base_acoustic_yaw_deg'])
        a=np.zeros((6,16384),np.float32);a[:,:min(r.shape[-1],16384)]=r[:,:16384];a/=record['rms']
        normal=rir_spectrum(torch.tensor(a)[None])[0].half()
        assert torch.equal(normal,scene['spectrum'][i].cpu()),'Cached preprocessing not reproduced'
        for arm in variants:
            x=a.copy()
            if arm=='direct_only':x[:,32:]=0
            elif arm=='direct_masked':x[:,:32]=0
            else:x[:,:7200]=0
            variants[arm].append(rir_spectrum(torch.tensor(x)[None])[0].half())
    return {k:torch.stack(v).cuda() for k,v in variants.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--modes',nargs='+',default=['acoustic','fusion']);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);protocol=json.loads((a.run/'protocol.json').read_text());models={};original={}
    for seed in protocol['seeds']:
      for mode in a.modes:
        model=StructureMatcher().cuda().eval();model.load_state_dict(torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True));models[(seed,mode)]=model
        old=json.loads((a.run/f'{mode}_{seed}.json').read_text());original[(seed,mode)]={(r['scene'],r['query']):r for r in old['val']['rows'] if r['arm']=='normal'}
    rows=[];mismatches=[];checked=0;waveforms=0
    with torch.no_grad():
      for sid in protocol['data']['scenes']['val']:
        scene=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));variants=audio_variants(a.root,a.cache,scene);waveforms+=len(scene['gt']);poses=scene['candidates'][:,:3]
        for (seed,mode),model in models.items():
            fp=full_descriptors(model,scene,True)
            for ids in torch.arange(len(scene['gt']),device='cuda').split(8):
                rgb=scene['rgb'][ids].float()/255;spec=scene['spectrum'][ids].float();v=model.visual(rgb)
                for arm in ('normal','audio_shuffle','audio_zero','direct_only','direct_masked','late_only','roll180'):
                    if arm=='normal':obs=spec
                    elif arm=='audio_shuffle':obs=scene['spectrum'][(ids+1)%len(scene['gt'])].float()
                    elif arm=='audio_zero':obs=torch.zeros_like(spec)
                    elif arm=='roll180':obs=torch.cat([spec[:,offset:offset+6].roll(3,1) for offset in (0,6,12)],1)
                    else:obs=variants[arm][ids].float()
                    acoustic=model.acoustic(obs);field=scores(model,v,acoustic,fp,scene,mode)
                    for j,q in enumerate(ids.tolist()):
                        gt=scene['gt'][q].clone()
                        if arm=='roll180' and mode=='acoustic':gt[2]=(gt[2]+math.pi)%(2*math.pi)
                        result=pose_statistics(field[j],poses,gt,diverse=arm=='normal')
                        if arm=='normal':
                            old=original[(seed,mode)][(sid,scene['query_ids'][q])];checked+=1
                            for key in ('joint','r1','fine','raw20','nll'):
                                if not np.isclose(old[key],result[key],atol=1e-5,rtol=1e-5):mismatches.append(dict(seed=seed,mode=mode,scene=sid,query=q,metric=key,old=old[key],new=result[key]))
                            indices,mass=support_nms(field[j],scene['valid'],poses)
                            d=poses[indices]-gt;dist=d[:,:2].norm(dim=-1);yaw=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs();ok=(dist<=1+1e-6)&(yaw<=math.pi/6+1e-6)
                            result.update(support_joint=float(ok[0]),support_r1=float(dist[0]<=1+1e-6),support_fine=float(dist[0]<=.1+1e-6),support20=float(ok.any()),support_candidate_count=len(indices))
                        if arm=='roll180' and mode=='acoustic':result['joint_against_unrotated_label']=pose_statistics(field[j],poses,scene['gt'][q],diverse=False)['joint']
                        rows.append(dict(scene=sid,query=scene['query_ids'][q],seed=seed,mode=mode,arm=arm,**result))
        (a.out/'predictions.json').write_text(json.dumps(rows,indent=2));print('SCENE_DONE',sid,flush=True)
    summary=[]
    for mode in a.modes:
      for arm in sorted({r['arm'] for r in rows}):
        rr=[r for r in rows if r['mode']==mode and r['arm']==arm];keys=['joint','r1','fine','raw20','nll']
        if arm=='normal':keys+=['diverse20','support_joint','support_r1','support_fine','support20']
        if arm=='roll180' and mode=='acoustic':keys+=['joint_against_unrotated_label']
        summary.append(dict(mode=mode,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in keys}))
    result=dict(summary=summary,normal_replay_checks=checked,normal_replay_mismatches=mismatches,raw_waveforms_reloaded=waveforms,
        windows='direct0:32; late7200:16384 samples (150–341.3ms); original RMS retained, no renormalization',
        rotation='3-mic cyclic permutation = 180deg rig yaw. Acoustic-only label rotates180deg; fusion keeps camera label and tests AV inconsistency.',
        support='Full posterior disk1m/yaw30deg mass, 3x3x3 local maxima then NMS1m/30deg, tau1; no fitted threshold')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
