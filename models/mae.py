# Masked Autoencoder from Meta.
# We get the pre trained model from transformers.
from transformers import AutoImageProcessor, ViTMAEModel
import torch
from torch import nn

# only works for imagenet ? 
class MAE(nn.Module):
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.mae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

    def forward(self, x):