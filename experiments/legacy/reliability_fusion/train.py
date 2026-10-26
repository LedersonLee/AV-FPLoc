"""Train constant/adaptive gates only; official test scores are never opened."""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from .prepare import REL
from .core import FEATURE_NAMES,metrics,prediction
from .model import Gate,loss


def load_scenes(root):
    out=root/REL;ready=read(out/'scores_ready.json');assert not ready['test_accessed']
    scenes=[]
    membership=read(root/'manifests/scene_splits_v1.json')
    for entry in ready['entries']:
        assert entry['split'] in ('train','val') and entry['scene_id'] in membership[entry['split']]
        folder=root/entry['directory'];assert sha(folder/'receipt.json')==entry['receipt_sha256']
        receipt=read(folder/'receipt.json')
        for name,digest in receipt['files'].items():assert sha(folder/name)==digest
        scenes.append(dict(**entry,arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in
            ('scores','features','target_expectations','truth','xy')}))
    assert not (set(s['scene_id'] for s in scenes if s['split']=='train')&set(s['scene_id'] for s in scenes if s['split']=='val'))
    return scenes


@torch.no_grad()
def evaluate(model,scenes,mean,std,device):
    model.eval();per_scene={};losses=[];weights_all=[]
    for scene in scenes:
        a=scene['arrays'];x=(np.asarray(a['features'])-mean)/std
        weights=model(torch.tensor(x,dtype=torch.float32,device=device)).cpu().numpy()
        poses=[];nll=[]
        for i,w in enumerate(weights):
            v,ac=a['scores'][i];poses.append(prediction(v,ac,a['xy'],w))
            # numpy float64 selection/NLL mirrors the frozen evaluation bank.
            s=((1-float(w))*v+float(w)*ac).ravel();m=float(s.max())
            nll.append(float(m+np.log(np.exp(s-m).sum())-np.dot([1-w,w],a['target_expectations'][i])))
        per_scene[scene['scene_id']]=metrics(poses,a['truth']);losses.append(np.mean(nll));weights_all.extend(weights.tolist())
    return dict(scene_macro={k:float(np.mean([r[k] for r in per_scene.values()])) for k in
        ('recall_01m','recall_05m','recall_1m','recall_1m_30deg')},nll=float(np.mean(losses)),
        per_scene=per_scene,weight_mean=float(np.mean(weights_all)),weight_p10=float(np.quantile(weights_all,.1)),
        weight_p90=float(np.quantile(weights_all,.9)))


def key(metrics):
    return (metrics['scene_macro']['recall_1m'],metrics['scene_macro']['recall_1m_30deg'],-metrics['nll'])


@torch.no_grad()
def figure(model,scenes,mean,std,device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(scenes),3,figsize=(14,4*len(scenes)),squeeze=False)
    for row,scene in enumerate(scenes):
        a=scene['arrays'];xy=a['xy'];v,ac=a['scores'][0];gt=a['truth'][0]
        w=float(model(torch.tensor((a['features'][:1]-mean)/std,dtype=torch.float32,device=device)).item())
        for ax,scores,title in zip(axes[row],(v,ac,(1-w)*v+w*ac),('Vision','Acoustic',f'Adaptive · acoustic w={w:.3f}')):
            im=ax.scatter(xy[:,0],xy[:,1],c=scores.max(1),s=8,cmap='coolwarm',marker='s')
            cell,yaw=np.unravel_index(scores.argmax(),scores.shape)
            for pose,color,label in [(gt,'dodgerblue','GT'),([*xy[cell],yaw*10],'limegreen','MAP')]:
                angle=np.deg2rad(pose[2]);ax.arrow(pose[0],pose[1],.6*np.cos(angle),.6*np.sin(angle),
                    width=.03,head_width=.18,color=color,label=label)
            ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(scene['scene_id']+'\n'+title);ax.legend()
            fig.colorbar(im,ax=ax,shrink=.6)
    fig.suptitle('Fixed first query per val scene · relative scores, not calibrated probabilities');fig.tight_layout()
    return fig


def run(root):
    root=Path(root);out=root/REL;config=read(out/'config.json')
    torch.set_num_threads(8);device=torch.device('cuda')
    scenes=load_scenes(root);train=[s for s in scenes if s['split']=='train'];val=[s for s in scenes if s['split']=='val']
    feature_values=np.concatenate([s['arrays']['features'] for s in train])
    mean,std=feature_values.mean(0),feature_values.std(0)
    std=np.where(std>1e-6,std,1.)
    atomic_json(out/'runs/normalization.json',dict(mean=mean.tolist(),std=std.tolist(),feature_names=FEATURE_NAMES,
        fit_split='train',train_scenes=[s['scene_id'] for s in train],test_accessed=False))
    results={}
    for mode in ('constant','adaptive'):
        folder=out/'runs'/mode;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():results[mode]=read(folder/'complete.json');continue
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        model=Gate(len(FEATURE_NAMES),constant=mode=='constant').to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        best=evaluate(model,val,mean,std,device);best_epoch=0;bad=0;history=[]
        def save(path,epoch):
            temporary=path.with_suffix('.tmp.pt');torch.save(dict(model=model.state_dict(),epoch=epoch,mode=mode,
                mean=mean,std=std,feature_names=FEATURE_NAMES,config_sha256=sha(out/'config.json'),
                scores_sha256=sha(out/'scores_ready.json'),test_accessed=False),temporary);temporary.replace(path)
        save(folder/'best.pt',0)
        tracker=None
        try:
            import wandb
            tracker=wandb.init(project='hssd-reliability-fusion',entity='aeonian-ulsan-national-institute-of-science-and-technology',
                name='pilot-v1-'+mode,dir=str(folder),config=dict(config,gate=mode),reinit=True)
            atomic_json(folder/'wandb.json',dict(url=tracker.url,id=tracker.id))
            n_total=sum(len(s['arrays']['features']) for s in train)
            for epoch in range(1,config['epochs']+1):
                model.train();batches=[]
                for scene in train:
                    ids=np.random.permutation(len(scene['arrays']['features']))
                    batches.extend((scene,ids[i:i+config['batch_size']]) for i in range(0,len(ids),config['batch_size']))
                random.shuffle(batches);losses=[]
                for scene,ids in batches:
                    a=scene['arrays'];x=torch.tensor((a['features'][ids]-mean)/std,dtype=torch.float32,device=device)
                    scores=torch.tensor(np.asarray(a['scores'][ids]),dtype=torch.float32,device=device).flatten(2)
                    expectation=torch.tensor(np.asarray(a['target_expectations'][ids]),dtype=torch.float32,device=device)
                    weight=model(x);objective=loss(weight,scores[:,0],scores[:,1],expectation)
                    # Equal expected contribution per scene, not per scene size.
                    balanced=objective*n_total/(len(train)*len(a['features']))
                    optimizer.zero_grad(set_to_none=True);balanced.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();losses.append(float(objective.detach()))
                validation=evaluate(model,val,mean,std,device)
                row=dict(epoch=epoch,train_nll=float(np.mean(losses)),val=validation);history.append(row)
                if key(validation)>key(best):best=validation;best_epoch=epoch;bad=0;save(folder/'best.pt',epoch)
                else:bad+=1
                save(folder/'last.pt',epoch);atomic_json(folder/'history.json',history)
                atomic_json(out/'training_status.json',dict(status='running',mode=mode,epoch=epoch,best_epoch=best_epoch,
                    val=validation,test_accessed=False))
                tracker.log(dict(epoch=epoch,train_nll=row['train_nll'],val_nll=validation['nll'],
                    **{'val/'+k:v for k,v in validation['scene_macro'].items()},weight_mean=validation['weight_mean'],
                    weight_p10=validation['weight_p10'],weight_p90=validation['weight_p90']))
                if epoch==1 or epoch%5==0:
                    fig=figure(model,val,mean,std,device);fig.savefig(folder/f'likelihood_{epoch:03d}.png',dpi=120)
                    tracker.log({'epoch':epoch,'val/likelihood':wandb.Image(fig)})
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                print(mode,epoch,row,flush=True)
                if bad>=config['patience']:break
            result=dict(status='complete',mode=mode,best_epoch=best_epoch,last_epoch=epoch,best_val=best,
                best_sha256=sha(folder/'best.pt'),test_accessed=False)
            atomic_json(folder/'complete.json',result);results[mode]=result
        finally:
            if tracker:tracker.finish()
    atomic_json(out/'runs/training_complete.json',dict(status='complete',results=results,
        best_sha256=sha(out/'runs/adaptive/best.pt'),config_sha256=sha(out/'config.json'),test_accessed=False))
    atomic_json(out/'training_status.json',dict(status='complete',test_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
