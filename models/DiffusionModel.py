import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

sys.path.append("../")




class DiffusionModel(nn.Module):
    """ Definition of  Tiny and Efficient Edge Detector
    model
    """

    def __init__(self, params, device="cuda"):
        super(DiffusionModel, self).__init__()
        self.params = params
        if self.params['dataset'] == 'cifar10':
            model_id = "google/ddpm-cifar10-32"
            self.diffnet = DDPMPipeline.from_pretrained(model_id) # pretrained on cifar10
        elif self.params['dataset'] == 'imagenet':
        
    def forward(self, x):
        x = self.diffnet(x)
        return x
        

if __name__ == "__main__":
