"""All audited observations, RGB and body-RIR only; labels separate from inputs."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class Observations(Dataset):
 def __init__(self,root,split,allow_partial=False):
  self.root=Path(root);manifest=json.loads((self.root/'manifest.json').read_text())
  if not allow_partial:assert manifest['ready'],'Full dataset audit/cache not complete'
  self.records=[r for r in manifest['records'] if r['split']==split];self.rows=[]
  for r in self.records:self.rows.extend((r['scene'],i) for i in range(r['queries']))
  assert self.rows
 def __len__(self):return len(self.rows)
 @lru_cache(maxsize=16)
 def scene(self,sid):
  p=self.root/sid;labels=dict(np.load(p/'labels.npz'));images=json.loads((p/'images.json').read_text());waves=np.load(p/'rir.npy',mmap_mode='r');return labels,images,waves
 def __getitem__(self,index):
  sid,i=self.rows[index];labels,images,waves=self.scene(sid)
  with Image.open(self.root/images[i]) as im:rgb=np.asarray(im.convert('RGB')).copy().transpose(2,0,1)
  return dict(rgb=torch.from_numpy(rgb),rir=torch.from_numpy(np.array(waves[i],copy=True)),target=torch.from_numpy(labels['target'][i].copy()),scene=sid,query_id=int(labels['query_ids'][i]))
