"""Execution smoke on one training scene; not a model evaluation experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .models import GeometryNet,masked_loss
from .train import evaluate
from .prepare_targets import read,sha


def main():
    p=argparse.ArgumentParser()
    for k in ['features','targets','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2)
    scene=read(a.targets/'protocol.json')['scenes']['train'][0]
    s=torch.load(a.features/f'{scene}.pt',map_location='cpu',weights_only=False)
    z=np.load(a.targets/f'{scene}.npz');s['depth']=torch.from_numpy(z['depth']);s['mask']=torch.from_numpy(z['mask'])
    rows=[]
    for arch in ['echo','spatial']:
      for head in ['regression','categorical']:
        torch.manual_seed(42);m=GeometryNet(arch,head).cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4)
        x=s['echo' if arch=='echo' else 'spatial'][:8].float().cuda()
        initial=[p.detach().clone() for p in m.encoder.parameters()]
        for step in range(2):
            m.train();mean,logits=m(x);loss=masked_loss(mean,logits,s['depth'][:8].cuda(),s['mask'][:8].cuda())
            opt.zero_grad();loss.backward();opt.step()
        changed=any(not torch.equal(p,v) for p,v in zip(m.encoder.parameters(),initial));assert changed
        metric,pred=evaluate(m,[s],'cuda');assert len(metric)==2 and len(pred)==2
        rows.append(dict(architecture=arch,head=head,encoder_changed=changed,finite_loss=bool(torch.isfinite(loss)),
                         parameters=sum(p.numel() for p in m.parameters()),metric_conditions=[r['condition'] for r in metric]))
    (a.out/'validation.json').write_text(json.dumps(dict(code_sha=a.code_sha,training_scene=scene,
        input_sha256=sha(a.features/f'{scene}.pt'),target_sha256=sha(a.targets/f'{scene}.npz'),
        scope='2-step execution check only, no scientific performance interpretation',checks=rows),indent=2))


if __name__=='__main__':main()
