import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from network_components import *
from network_initializer import ResNet


class pspnet(nn.Module):
    """
    PSP Net implementation based on BottleNeck typed ResNet
    """
    def __init__(self, model_type):
        """
        Args:
          - model_type: Now, we can choose "resnet18", "resnet34", "resnet50",
                        "resnet101" and "resnet152" as a feature extractor.
        """
        super(pspnet, self).__init__()
        self.extractor = ResNet(model_type)
        self.extractor.init_pretrained()
        self.pyramid_pooling = PyramidPoolingModule(1024, sizes=(1, 2, 3, 6))



class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()

        self.pyramid_features = nn.ModuleList(
            [make_pool_module(in_channels, sizes[i], len(sizes))
             for i in range(len(sizes))])

    def make_pool_module(self, in_channels, size, pyramid_depth):
        pooling = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, int(in_channels / pyramid_depth),
                         kernel_size=1, bias=False)
        return nn.Sequential(pooling, conv)

    def forward(self, inputs):
        features = [F.interpolate(feature(inputs),
                                  size=input.size()[2:],
                                  mode="bilinear")
                    for feature in self.pyramid_features]
        features.append(inputs)

        return torch.cat(features, dim=1)

class UpSample(nn.Module)
