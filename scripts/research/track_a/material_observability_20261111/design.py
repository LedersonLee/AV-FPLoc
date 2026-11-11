"""Procedural meshes and exact coefficient design; no synthetic acoustic signals."""
import copy,json
from pathlib import Path
POSES=[dict(id='single2',kind='single',x=.17,z=2.,split='fit'),dict(id='single6',kind='single',x=.17,z=6.,split='evaluation'),dict(id='room_near',kind='room',x=.17,z=2.,split='evaluation'),dict(id='room_far',kind='room',x=1.1,z=6.,split='evaluation'),dict(id='corridor_near',kind='corridor',x=.17,z=2.,split='evaluation'),dict(id='corridor_far',kind='corridor',x=-.5,z=6.,split='evaluation')]
def mesh(pose,state,path):
 kind=pose['kind'];polygon=([[-12,0],[12,0]] if kind=='single' else [[-4,0],[4,0],[4,8],[-4,8],[-4,0]] if kind=='room' else [[-1.2,0],[1.2,0],[1.2,18],[-1.2,18],[-1.2,0]])
 segments=list(zip(polygon[:-1],polygon[1:]));verts=[];faces=[];kept=[]
 def quad(v,label):
  o=len(verts);verts.extend(v);faces.extend([[o+i for i in f]+[label] for f in [[0,1,2],[0,2,3],[2,1,0],[3,2,0]]])
 for i,(a,b) in enumerate(segments):
  if i==0 and state=='removed':continue
  w=-1 if i==0 and state=='moved' else 0;a=[a[0],a[1]+w];b=[b[0],b[1]+w];kept.append([a,b]);lo,hi=(-8,12) if kind=='single' else (0,3)
  quad([[a[0],lo,a[1]],[b[0],lo,b[1]],[b[0],hi,b[1]],[a[0],hi,a[1]]],0)
 if kind=='single':quad([[-12,-20,-12],[12,-20,-12],[12,-20,24],[-12,-20,24]],1)
 else:
  width,end=(4,8) if kind=='room' else (1.2,18)
  for y in [0,3]:quad([[-width,y,0],[width,y,0],[width,y,end],[-width,y,end]],1)
 path.write_text(f'{len(verts)} {len(faces)}\n'+'\n'.join(' '.join(map(str,v)) for v in verts+faces)+'\n');return dict(segments=kept,source=[pose['x'],1.5,pose['z']],state=state,vertices=verts,faces=faces)
def material(default,absorption,scattering,profile):
 # Symmetric low/high profiles have equal arithmetic mean at the four log-spaced anchors.
 frequencies=[83.910034,369.251007,1624.911255,7150.516602];delta=min(absorption,1-absorption)*.8
 shape=[1,1,-1,-1] if profile=='low' else [-1,-1,1,1] if profile=='high' else [0,0,0,0]
 values=[absorption+delta*v for v in shape];mat=copy.deepcopy(default);mat['labels']=['audit_surface'];mat['name']='Controlled surface';mat['absorption']=sum(([f,v] for f,v in zip(frequencies,values)),[]);mat['scattering']=[20.,scattering,20000.,scattering];mat['transmission']=[20.,0.,20000.,0.]
 bg=copy.deepcopy(default);bg['labels']=['fixed_background'];bg['name']='Fixed background';return dict(materials=[mat,bg])
