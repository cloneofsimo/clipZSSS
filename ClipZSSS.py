import torch
from CAMERAS import CAMERAS
import torch
import torch.nn.functional as F
from torch import nn

class ZSSS(nn.Module):
    
    def __init__(self, modelType = "RN50"):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(modelType, device=device, jit = False)
        model.to(device)
        self.model = model
        self.semantic_class = 
    
    def forward(self, )



if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
