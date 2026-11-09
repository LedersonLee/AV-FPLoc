"""Training fit vs novel-scene failure; paired controls with frozen pilot weights."""
import argparse,json,math,time,sys
from pathlib import Path
import numpy as np
import torch
from track_b.scripts.pilot import materialize,metrics,jread
from track_b.models.dense import DenseScorer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2);opened=[]
    def audit(event,values):
        if event=='open' and isinstance(values[0],str):
            path=values[0]
            if '/RIR/FP/' in path:raise RuntimeError('Reference RIR access forbidden')
            if '/RIR/' in path:opened.append(path)
    sys.addaudithook(audit)
    protocol=jread(args.pilot/'protocol.json');rows=[];collision=[];timings=[]
    models={}
    for seed in protocol['seeds']:
      for mode in protocol['modes']:
        m=DenseScorer(mode=mode).cuda().eval();m.load_state_dict(torch.load(args.pilot/f'{mode}_{seed}.pt',weights_only=True));models[(seed,mode)]=m
    for split in ('train','val'):
      for sid in protocol[split]:
        scene=materialize(args.root,sid,split,16);p=scene['p']
        for seed in protocol['seeds']:
          with torch.no_grad():
            for rgb,rir,gt,qid in scene['data']:
              scores={mode:models[(seed,mode)](rgb,rir,scene['fp'],scene['cal'],p)[0] for mode in protocol['modes']}
              def z(s):return (s-s.mean())/s.std().clamp_min(1e-6)
              scores['fixed_fusion']=.85*z(scores['visual'])+.15*z(scores['acoustic'])
              for mode,s in scores.items():
                # Training diagnostics need no diverse NMS; metrics records exact MAP + raw oracle.
                d=p-gt;dist=d[:,:2].norm(dim=-1);angle=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs();ok=(dist<=1)&(angle<=math.pi/6+1e-6);idx=s.argmax()
                rows.append(dict(split=split,scene=sid,seed=seed,mode=mode,query=qid,joint=float(ok[idx]),r1=float(dist[idx]<=1),fine=float(dist[idx]<=.1),raw20=float(ok[s.topk(20).indices].any())))
              if seed==42:
                d=p-gt;dist=d[:,:2].norm(dim=-1);angle=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs();ok=(dist<=1)&(angle<=math.pi/6+1e-6)
                rows.append(dict(split=split,scene=sid,seed=seed,mode='uniform_random_expectation',query=qid,joint=float(ok.float().mean()),r1=float((dist<=1).float().mean()),fine=float((dist<=.1).float().mean()),raw20=None))
            # Quantized random initial FP features: a diagnostic of local patch collisions.
            if seed==42:
              m=models[(seed,'joint')];rgb,rir,_,_=scene['data'][0];v,a,fp=m.encode(rgb,rir,scene['fp'])
              captured=[]
              h=m.fp_query.register_forward_hook(lambda module,inp,out:captured.append(out[0].detach()))
              for chunk in p.split(512): m.score_chunk(v,a,fp,chunk,scene['cal'])
              h.remove();q=torch.cat(captured).cpu().numpy();unique,counts=np.unique(np.round(q,5),axis=0,return_counts=True)
              collision.append(dict(scene=sid,split=split,poses=len(p),unique_fp_queries=len(unique),largest_equal_group=int(counts.max()),rounding_decimals=5,interpretation='Rounded FP queries only; near-equality does not prove exact prediction equivalence'))
              for _ in range(3):m(rgb,rir,scene['fp'],scene['cal'],p)
              torch.cuda.synchronize();start=time.perf_counter()
              for _ in range(10):m(rgb,rir,scene['fp'],scene['cal'],p)
              torch.cuda.synchronize();timings.append(dict(scene=sid,poses=len(p),warm_ms=(time.perf_counter()-start)*100,scope='RGB/RIR/FP encoders and exhaustive scorer; excludes file I/O, preprocessing and Top20'))
    summary=[]
    for split in ('train','val'):
      for mode in protocol['modes']+['fixed_fusion','uniform_random_expectation']:
        rr=[r for r in rows if r['split']==split and r['mode']==mode]
        summary.append(dict(split=split,mode=mode,**{k:float(np.mean([r[k] for r in rr])) for k in ('joint','r1','fine')}))
    args.out.write_text(json.dumps(dict(summary=summary,rows=rows,collisions=collision,timings=timings,io_trace=dict(reference_reads=0,observed_reads=sum('/RIR/Observed/' in x for x in opened),paths=sorted(set(opened)))),indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
