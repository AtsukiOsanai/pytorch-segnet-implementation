import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from network_components import *
from network_initializer import ResNet


class pspnet_pure(nn.Module):
    """
    PSP Net implementation based on Pure ResNet
    """
    def __init__(self, model_type, n_classes):
        """
        Args:
          - model_type: Now, we can choose "resnet18", "resnet34", "resnet50",
                        "resnet101" and "resnet152" as a feature extractor.
          - n_classes: The number of classes
        """
        super(pspnet, self).__init__()
        self.extractor = ResNet(model_type)
        self.extractor.init_pretrained()
        self.pyramid_pooling = PyramidPoolingModule(2048, sizes=(1, 2, 3, 6))
        self.upsample1 = UpSample(4096, 512)
        self.upsample2 = UpSample(512, 64)
        self.classifier = nn.Conv2d(64, n_classes, 1)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)

        # auxiliary
        self.aux1 = Conv2dBatchNormRelu(1024, 256, 3)
        self.aux2 = Conv2dBatchNormRelu(256, n_classes, 3)


    def forward(self, inputs):
        aux, outputs = self.extractor(inputs, output_aux=True)
        outputs = self.pyramid_pooling(outputs)
        outputs = self.upsample1(outputs)
        outputs = self.dropout(outputs)
        outputs = self.upsample2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)

        outputs = F.interpolate(outpus, size=inputs.shape[2:], mode="bilinear")

        if self.training:
            h, w = 4 * aux.shape[2], 4 * aux.shape[3]
            aux_outputs = self.aux1(aux)
            aux_outputs = F.interpolate(aux_outputs, size=(h, w), mode="bilinear")
            h, w = 4 * aux_outputs.shape[2], 4 * aux_outputs.shape[3]
            aux_outputs = self.aux2(aux_outputs)
            aux_outputs = F.interpolate(aux_outputs, size=(h, w), mode="bilinear")

            return outputs, aux_outputs

        else:
            return outputs

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()

        self.pyramid_features = nn.ModuleList(
            [make_pool_module(in_channels, sizes[i], len(sizes))
             for i in range(len(sizes))])

    def make_pool_module(self, in_channels, size, pyramid_depth):
        # TODO check whether in_channels % pyramid_depth = 0 or not
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

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.conv = Conv2dBatchNormRelu(in_channels, out_channels,
                                        k_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, inputs):
        h, w =  2 * inputs.size(2), 2 * inputs.size(3)
        outputs = F.interpolate(
            self.conv(inputs), size=(h, w), mode="bilinear")

        return outputs
