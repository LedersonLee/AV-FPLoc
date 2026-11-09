"""Keep expanded scene tensors on CPU, transferring only the requested scene."""
import torch

class SceneSequence:
    def __init__(self, scenes, device='cuda'):
        self.scenes=scenes
        self.device=device

    def __len__(self):return len(self.scenes)

    def __getitem__(self, index):
        scene=self.scenes[index]
        return {k:v.to(self.device) if isinstance(v,torch.Tensor) and k!='valid' else v
                for k,v in scene.items()}

    def __iter__(self):
        for i in range(len(self)):yield self[i]
