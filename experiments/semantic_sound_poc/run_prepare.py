"""Freeze assets and geometry-only source proposals; no GT-based query filtering."""
import argparse, json, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from matplotlib.path import Path as Polygon

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));args=a.parse_args();out=args.out;c=json.loads(args.config.read_text());r=Path(c['root']);w=Path(c['workspace']);s=r/'source/hssd-hab';sid=c['scene'];out.mkdir(parents=True,exist_ok=True)
 split=json.loads((r/'manifests/scene_splits_v1.json').read_text());assert sid in split['train'];assert sid.split('_')[0] not in {x.split('_')[0] for x in split['test']}
 sem=s/'semantics/scenes'/f'{sid}.semantic_config.json';regions=json.loads(sem.read_text())['region_annotations'];mp=r/'2DFP'/sid/'map.png';meta=json.loads(mp.with_name('floorplan_metadata.json').read_text());assert sha(sem)==meta['semantic_sha256'];assert meta['raster_view']=='+Y top-down: +X right, +Z down';free=np.array(Image.open(mp))==255;dist=distance_transform_edt(free)*meta['map_spacing_m'];b=meta['world_bounds_m'];origin=np.array([b['x_min'],b['z_min']]);step=meta['map_spacing_m'];pools=[];rng=np.random.default_rng(c['geometry_seed']);audit=[]
 for i,reg in enumerate(regions):
  poly=np.array(reg['poly_loop'])[:,[0,2]];lo=poly.min(0);hi=poly.max(0);x,z=np.meshgrid(np.arange(lo[0],hi[0],c['grid_spacing_m']),np.arange(lo[1],hi[1],c['grid_spacing_m']));pts=np.stack([x.ravel(),z.ravel()],1);mask=Polygon(poly).contains_points(pts)
  for off in [[.15,0],[-.15,0],[0,.15],[0,-.15]]:mask &= Polygon(poly).contains_points(pts+off)
  pix=np.rint((pts-origin)/step).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<free.shape[1])&(pix[:,1]>=0)&(pix[:,1]<free.shape[0]);mask &= inside;safe=np.zeros(len(pts),bool);safe[inside]=dist[pix[inside,1],pix[inside,0]]>=c['clearance_m'];mask &=safe;pts=pts[mask];rng.shuffle(pts);pools.append(dict(room=i,label=reg['label'],points_xz=pts.tolist()));audit.append(dict(room=i,label=reg['label'],polygon_area_m2=float(abs(np.dot(poly[:,0],np.roll(poly[:,1],1))-np.dot(poly[:,1],np.roll(poly[:,0],1)))/2),valid_grid_points=len(pts)))
 # Compare semantic indoor support to raster; map is itself semantic-supported, not independent ground truth.
 yy,xx=np.where(free);metric=np.stack([xx,yy],1)*step+origin;covered=np.zeros(len(metric),bool)
 for reg in regions:covered |= Polygon(np.array(reg['poly_loop'])[:,[0,2]]).contains_points(metric,radius=.02)
 comp,n=label(free);sizes=np.bincount(comp.ravel())[1:];alignment=dict(semantic_hash_matches_export=True,free_pixels=len(metric),fraction_free_in_semantic_polygons=float(covered.mean()),raster_components=int(n),largest_component_fraction=float(sizes.max()/sizes.sum()),pixel_roundtrip_max_error_m=float(np.max(abs((np.rint((metric-origin)/step)*step+origin)-metric))),origin_xz=origin.tolist(),spacing_m=step,rooms=audit)
 cand=w/'experiments/directional_structure_20261109/candidates'/sid/'pure_visual.npz';v=np.load(cand);rows=np.flatnonzero(v['vs'][:,0]-v['vs'][:,1]<=c['ambiguity_max_top2_log_gap']);np.savez_compressed(out/'inference_candidates.npz',poses=v['pred'][rows],scores=v['vs'][rows],query_ids=v['query_ids'][rows],original_rows=rows)
 truthpath=w/'experiments/directional_tribranch_20261109/data'/sid/'observations.npz';o=np.load(truthpath);np.testing.assert_array_equal(v['query_ids'],o['query_ids']);np.savez_compressed(out/'evaluation_truth.npz',truth=o['truth'][rows],query_ids=v['query_ids'][rows])
 # Collect real image paths from original and replacement manifests.
 rgb=r/'UnLoc/RGB'/sid;rec={int(q['sample_index']):dict(q,image_root=str(rgb)) for q in json.loads((rgb/'pilot_manifest.json').read_text())['records'] if q['accepted']};rep=r/'UnLoc/furnished_depth_qa_v2_nohit_only/replacements'/sid/'replacement_manifest.json'
 if rep.exists():
  for q in json.loads(rep.read_text())['records']:
   if q.get('accepted'):rec[int(q['sample_index'])]=dict(q,image_root=str(rep.parent))
 images={str(int(qid)):str(Path(rec[int(qid)]['image_root'])/rec[int(qid)]['image']) for qid in v['query_ids'][rows]};write(out/'images.json',images)
 # Rewrite existing minimal registry relative paths to absolute native paths; original remains read-only.
 oldreg=r/'RIR/Geometry'/sid/'furnished.scene_dataset_config.json';registry=json.loads(oldreg.read_text())
 for key in ['stages','objects','scene_instances']:
  for ext,paths in registry[key].get('paths',{}).items():registry[key]['paths'][ext]=[str((oldreg.parent/p).resolve()) for p in paths]
 registry['semantic_scene_descriptor_instances']={k:str((oldreg.parent/p).resolve()) for k,p in registry.get('semantic_scene_descriptor_instances',{}).items()};write(out/'furnished.scene_dataset_config.json',registry)
 paths=[sem,mp,mp.with_name('floorplan_metadata.json'),cand,truthpath,r/'manifests/scene_splits_v1.json',s/'scenes'/f'{sid}.scene_instance.json',s/'stages'/f'{sid}.stage_config.json',s/'stages'/f'{sid}.glb',oldreg,args.config,w/'realW_exp/analysis/channel_map/uma8_channel_map.json'];write(out/'input_manifest.json',dict(files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],scene=sid,queries_before=128,queries_after=len(rows),visual_prior='actual cached pure-visual NMS20 raw scores; softmax T=1 conditional, not calibrated posterior',exposure='Train scene; visual teacher Train116/Val19 and repeated development upstream; Exploratory',images_found=sum(Path(p).exists() for p in images.values()),test_observations_read=False));write(out/'source_proposals.json',pools);write(out/'alignment.json',alignment);write(out/'config.json',c);print(json.dumps(dict(scene=sid,N=len(rows),alignment=alignment,images_found=sum(Path(p).exists() for p in images.values()))))
if __name__=='__main__':main()
