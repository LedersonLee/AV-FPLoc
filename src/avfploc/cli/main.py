"""Small portable commands; dataset rendering and model training are explicit."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(prog='avfploc')
    sub=p.add_subparsers(dest='command',required=True)
    fusion=sub.add_parser('fuse',help='Fuse aligned portable Vision/Acoustic NPZ banks')
    fusion.add_argument('--vision',type=Path,required=True);fusion.add_argument('--acoustic',type=Path,required=True)
    fusion.add_argument('--output',type=Path,required=True);fusion.add_argument('--weight',type=float,default=.5)
    fusion.add_argument('--checkpoint',type=Path);fusion.add_argument('--schema',choices=['core14','processed58','legacy17'],default='processed58')
    fusion.add_argument('--device',default='cpu')
    train=sub.add_parser('train-fusion',help='Train on explicit disjoint Train/Val manifests')
    train.add_argument('--manifest',type=Path,required=True);train.add_argument('--output',type=Path,required=True)
    train.add_argument('--epochs',type=int,default=17);train.add_argument('--schema',choices=['core14','processed58','legacy17'],default='processed58')
    train.add_argument('--device',default='cpu');train.add_argument('--temperature',type=float,default=1.)
    train.add_argument('--objective',choices=['nll','margin_xy','margin_joint'],default='nll')
    vision=sub.add_parser('vision',help='RGB images -> depth and aligned score bank')
    vision.add_argument('--inputs',type=Path,required=True);vision.add_argument('--output',type=Path,required=True)
    vision.add_argument('--checkpoint',type=Path,required=True);vision.add_argument('--unloc-root',type=Path,required=True)
    vision.add_argument('--device',default='cuda')
    audio=sub.add_parser('acoustic',help='Prepared E/C/N features -> aligned score bank')
    audio.add_argument('--inputs',type=Path,required=True);audio.add_argument('--output',type=Path,required=True)
    audio.add_argument('--checkpoint',type=Path,required=True);audio.add_argument('--device',default='cuda')
    evaluation=sub.add_parser('evaluate')
    evaluation.add_argument('--predictions',type=Path,required=True);evaluation.add_argument('--truth',type=Path,required=True)
    args=p.parse_args()
    if hasattr(args,'output') and args.output.exists():p.error('Output already exists; choose a new destination')
    if args.command=='train-fusion':
        from avfploc.fusion.train import run
        print(json.dumps(run(args.manifest,args.output,epochs=args.epochs,schema=args.schema,device=args.device,temperature=args.temperature,objective=args.objective)))
        return
    if args.command=='evaluate':
        from avfploc.evaluation.metrics import compare
        with np.load(args.predictions,allow_pickle=False) as v, np.load(args.truth,allow_pickle=False) as t:
            if not np.array_equal(v['query_ids'],t['query_ids']):raise ValueError('Query order differs')
            print(json.dumps(compare(v['poses'],t['poses'],v['fixed01_poses'] if 'fixed01_poses' in v else None),indent=2))
        return
    if args.command=='fuse':
        from avfploc.data.banks import load,aligned
        from avfploc.fusion.fixed import fuse,decode
        v,a=load(args.vision),load(args.acoustic);aligned(v,a)
        gate=None
        if args.checkpoint:
            from avfploc.fusion.infer import load_gate,predict
            gate=load_gate(args.checkpoint,args.device)
        scores=[];poses=[];weights=[];fixed=[]
        for i,(vs,ac) in enumerate(zip(v['scores'],a['scores'])):
            if gate is not None:
                s,w=predict(vs,ac,v['xy'],v['depth'][i],gate,args.schema,v.get('ray_bank'))
            else:s,w=fuse(vs,ac,args.weight),args.weight
            scores.append(s);weights.append(w);poses.append(decode(s,v['xy']));fixed.append(decode(fuse(vs,ac,.1),v['xy']))
        result=dict(scores=scores,poses=poses,weights=weights,xy=v['xy'],query_ids=v['query_ids'],yaw_deg=v['yaw_deg'],fixed01_poses=fixed)
    elif args.command=='vision':
        from avfploc.vision.infer import rgb_depth,score
        with np.load(args.inputs,allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
        images=[str((args.inputs.parent/str(x)).resolve()) for x in inputs['images']]
        depth,scale=rgb_depth(args.checkpoint.resolve(),args.unloc_root,images,args.device)
        result=dict(scores=score(depth,scale,inputs['ray_bank']),depth=depth,scale=scale,ray_bank=inputs['ray_bank'],xy=inputs['xy'],query_ids=inputs['query_ids'],yaw_deg=np.arange(36)*10)
    elif args.command=='acoustic':
        from avfploc.acoustic.infer import encode,score
        with np.load(args.inputs,allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
        obs=encode(args.checkpoint,inputs['observed_energy'],inputs['observed_coherence'],inputs.get('observed_envelope'),args.device)
        cand=encode(args.checkpoint,inputs['candidate_energy'],inputs['candidate_coherence'],inputs.get('candidate_envelope'),args.device)
        if len(cand)!=len(inputs['xy'])*36:raise ValueError('Candidates must be cell-major, yaw-minor (36 headings)')
        result=dict(scores=score(obs,cand.reshape(len(inputs['xy']),36,-1)),xy=inputs['xy'],query_ids=inputs['query_ids'],yaw_deg=np.arange(36)*10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:np.savez_compressed(f,**result)
    # Re-read serialized identity/shape before a downstream process consumes it.
    if 'scores' in result:
        from avfploc.data.banks import load
        load(args.output)
    print(args.output)


if __name__=='__main__':main()
