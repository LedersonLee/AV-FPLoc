"""Post-fit error by supplied semantic room label and explicit geometry strata."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_targets import read,sha
from .summarize import paired


def contains(point,polygon):
    x,y=point;inside=False
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def main():
    p=argparse.ArgumentParser()
    for k in ['root','study','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    protocol=read(a.study/'representation_v1/protocol.json');labels={};sources=[]
    for scene in protocol['scene_ids']['val']:
        semantic=a.root/f'source/hssd-hab/semantics/scenes/{scene}.semantic_config.json'
        regions=read(semantic)['region_annotations'];z=np.load(a.study/'targets_v3'/f'{scene}.npz');refs=read(a.study/'targets_v3'/f'{scene}_queries.json')
        source=[]
        for i,pose in enumerate(z['poses']):
            hit={r['label'] for r in regions if contains(pose[:2],np.asarray(r['poly_loop'])[:,[0,2]])}
            room=next(iter(hit)) if len(hit)==1 else 'ambiguous' if hit else 'unassigned'
            ranges=z['depth'][i,z['mask'][i]];assert len(ranges)>0
            strata=['room:'+room,'near_wall:<1m' if ranges.min()<1 else 'near_wall:>=1m',
                'range_variation:>=4m' if ranges.max()-ranges.min()>=4 else 'range_variation:<4m',
                'valid_directions:<75%' if z['mask'][i].mean()<.75 else 'valid_directions:>=75%']
            labels[(scene,refs[i]['query_id'])]=strata;source.append(dict(query=refs[i]['query_id'],strata=strata))
        sources.append(dict(scene=scene,semantic_sha256=sha(semantic),labels=source))
    groups={};counts={}
    def add(arm,metric,scene,query,seed,value):
        for category in labels[(scene,query)]:
            key=(arm,metric,category);groups.setdefault(key,{}).setdefault((scene,seed),[]).append(float(value))
            counts.setdefault(key,set()).add((scene,query))
    for r in read(a.study/'representation_v1/registry.json'):
        for scene in protocol['scene_ids']['val']:
            z=np.load(a.study/'representation_v1'/r['run']/f'{scene}__normal.npz');refs=read(a.study/'targets_v3'/f'{scene}_queries.json')
            for i,q in enumerate(refs):
                e=z['mean'][i]-z['target'][i];m=z['mask'][i]
                add(r['architecture']+'/'+r['head'],'depth_mae_m',scene,q['query_id'],r['seed'],abs(e[m]).mean())
                add(r['architecture']+'/'+r['head'],'depth_rmse_m',scene,q['query_id'],r['seed'],np.sqrt(np.mean(e[m]**2)))
    for r in read(a.study/'transfer_v1/registry.json'):
        if r['arm']=='visual':continue
        for x in read(a.study/'transfer_v1'/f"{r['arm']}_{r['seed']}"/'metrics.json'):
            if x['condition']!='fixed_fusion':continue
            for metric in ['joint','r1','fine','nll']:
                add('transfer/'+r['arm'],metric,x['scene'],x['query'],r['seed'],x[metric])
    summary=[]
    for (arm,metric,category),values in sorted(groups.items()):
        scenes={}
        for (scene,seed),v in values.items():scenes.setdefault(scene,[]).append(np.mean(v))
        estimate=paired({s:np.mean(v) for s,v in scenes.items()})
        if estimate['independent_families']<3:estimate['ci95']=None
        summary.append(dict(arm=arm,metric=metric,category=category,queries=len(counts[(arm,metric,category)]),**estimate))
    (a.out/'summary.json').write_text(json.dumps(dict(code_sha=a.code_sha,definition='Semantic polygon labels (ambiguous overlaps retained), nearest valid stage hit, angular range span and target coverage; diagnostic thresholds1m/4m/75%, not selected for model changes',
        caveat='These are observed structural/context strata, not randomized room-type effects. RMSE is within query then scene/family averaged; not the primary per-scene pooled-ray RMSE. Intervals suppressed below3 families.',summary=summary,sources=sources),indent=2))


if __name__=='__main__':main()
