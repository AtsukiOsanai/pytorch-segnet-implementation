import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from network_components import *


class pspnet(nn.Module):
    """
    PSP Net implementation based on BottleNeck typed ResNet
    """
    def __init__(self, model):
        super(pspnet, self).__init__()



class pool_conv(nn.Module):
    def __init__(self, k_pool,
                 conv_in_channels, conv_out_channels):
        super(pool_conv, self).__init__()

        self.pool = nn.AvgPool2d(k_pool)
        self.conv = Conv2dBatchNormRelu(conv_in_channels, conv_out_channels,
                                        k_size=1, padding=0, stride=1,
                                        bias=False)
    def forward(self, inputs):
        outputs = self.pool(inputs)
        outputs = self.conv(outputs)
        outputs = nn.functional.interpolate(
            outputs, size=inputs.size()[2:], mode='bilinear')

        return outputs
