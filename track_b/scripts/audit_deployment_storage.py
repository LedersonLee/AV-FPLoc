"""Measure geometry setup and isolated reference-file storage, without loading RIRs."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from track_b.data.geometry import candidate_bank
from track_b.data.rays import boundary_rays

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();protocol=json.loads((a.cache/'protocol.json').read_text());rows=[]
 for sid in protocol['scenes']['val']:
  meta_path=a.root/f'2DFP/{sid}/floorplan_metadata.json';map_path=a.root/f'2DFP/{sid}/map.png';meta=json.loads(meta_path.read_text());free=np.array(Image.open(map_path))==255;origin=[meta['world_bounds_m'][k] for k in ('x_min','z_min')];spacing=meta['map_spacing_m'];start=time.perf_counter();poses,valid=candidate_bank(free,spacing,origin,.25,36);ranges,hits=boundary_rays(free,spacing,origin,poses[::36,:2],36);seconds=time.perf_counter()-start
  # Set membership tolerance only for bank-overlap audit, not localization metrics.
  keys={(round(float(x),4),round(float(z),4)) for x,z in poses[::36,:2]};groups=collections.defaultdict(lambda:dict(positions=0,npy_files=0,npy_bytes=0,missing_expected_yaws=0,matched_positions=0,matched_npy_bytes=0));matched=set();sidecar_digest=hashlib.sha256();inventory_start=time.perf_counter()
  records=a.root/f'RIR/FP/val/{sid}/records'
  for folder in sorted(records.iterdir()):
   if not folder.is_dir():continue
   paths={p.name:p for p in folder.iterdir()};side=paths.get('base_acoustic_yaw_000.json')
   if side is None:continue
   raw=side.read_bytes();sidecar_digest.update(hashlib.sha256(raw).digest());d=json.loads(raw);assert d['split']=='val' and d['scene_id']==sid and d['role']=='FP';key=(round(float(d['centre_xyz_m'][0]),4),round(float(d['centre_xyz_m'][2]),4));overlap=key in keys;group=groups[d.get('version','unknown')+'|'+d.get('geometry_kind','unknown')];group['positions']+=1
   files=[p for n,p in paths.items() if n.endswith('.npy')];sizes=sum(p.stat().st_size for p in files);group['npy_files']+=len(files);group['npy_bytes']+=sizes;group['missing_expected_yaws']+=sum(f'base_acoustic_yaw_{yaw:03d}.npy' not in paths for yaw in (0,10,20,30,40,50))
   if overlap:group['matched_positions']+=1;group['matched_npy_bytes']+=sizes;matched.add(key)
  row=dict(scene=sid,candidates=len(poses),valid_xy=len(keys),geometry_setup_seconds=seconds,geometry_setup_scope='CPU bank + boundary rays only; no encoder/RIR simulation; one run, no cold filesystem claim',expanded_candidate_float32_bytes=len(poses)*35*4,raw_range_and_hit_arrays_bytes=ranges.nbytes+hits.nbytes,xy_fp_mask_bytes=valid.nbytes,floorplan_png_bytes=map_path.stat().st_size,learned_descriptor_float32_bytes_per_encoder=len(poses)*16*4,reference_groups=dict(groups),unique_reference_xy_matches=len(matched),xy_overlap_fraction=len(matched)/len(keys),matching_tolerance='rounded native X/Z to 4 decimals; yaw files counted, not loaded or independently waveform-verified',reference_metadata_inventory_seconds=time.perf_counter()-inventory_start,ordered_sidecar_sha256_digest=sidecar_digest.hexdigest());rows.append(row);a.out.write_text(json.dumps(dict(rows=rows,status='running',limits='Reference storage audit only; no FP-RIR used by proposed scorer; no matched B2 accuracy or historical generation time inferred.'),indent=2));print(json.dumps(row),flush=True)
 result=dict(rows=rows,status='complete',limits='Reference storage audit only; no FP-RIR used by proposed scorer; no matched B2 accuracy or historical generation time inferred. Observed query RIR training generation/storage is not eliminated. Metadata inventory latency is NOT reference generation latency.');a.out.write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
