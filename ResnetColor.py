from typing import Tuple, List, Optional, Union
import torchvision

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class ResnetColor(nn.Module):

    def __init__(self):
        super().__init__()

        # resnet101 가져오기
        resnet101 = torchvision.models.resnet101(pretrained=False)
        children = list(resnet101.children())
        features = children[:-3]
        
        # resnet101 output -> conv1
        self.features = nn.Sequential(*features)
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=4, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = x.permute(0,3,2,1)
        x = x.reshape(-1,4)
        return x    