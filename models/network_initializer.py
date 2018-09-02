import torcn.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models


from network_components import *


resnet_config = {"resnet18": {"block": BasicBlock,
                              "layers": [2, 2, 2, 2],
                              "model_url": "https://download.pytorch.org/models/resnet18-5c106cde.pth"},
                 "resnet34": {"block": BasicBlock,
                              "layers": [3, 4, 6, 5],
                              "model_url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth"},
                 "resnet50":  {"block": Bottleneck,
                               "layers": [3, 4, 6, 3],
                               "model_url": "https://download.pytorch.org/models/resnet50-19c8e357.pth"},
                 "resnet101": {"block": Bottleneck,
                               "layers": [3, 4, 23, 3],
                               "model_url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"},
                 "resnet152": {"block": Bottleneck,
                               "layers": [3, 8, 36, 3],
                               "model_url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth"},
                }

class ResNet(nn.Module):
    """
    ResNet parent class as a feature extractor
    """
    def __init__(self, resnet_type):
        super(ResNet, self).__init__()
        self.block = resnet_config[resnet_type]["block"]
        self.layers = resnet_config[resnet_type]["layers"]
        self.model_url = resnet_config[resnet_type]["model_url"]

        self.conv_bn_relu = Conv2dBatchNormRelu(3, 64, 7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # TODO Fix the channel mismatch
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], down_sample=True)
        self.layer3 = self._make_layer(block, 512, 256, layers[2], down_sample=True)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], down_sample=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channels, aux_channels, num_layers, down_sample=False):
        layers = []
        layers.append(block(in_channels, aux_channels, down_sample))
        for i in range(1, num_layers):
            layers.append(block(aux_channels * block.expansion, aux_channels))

        return nn.Sequential(*layers)

    def init_pretrained(self):
        pretrained_params = model_zoo.load_url(self.model_url)


        blocks = [self.conv_bn_relu,
                  self.layer1,
                  self.layer2,
                  self.layer3,
                  self.layer4]
        target_layers = []
        for idx, block in enumerate(model.layer1):
            for unit in list(block.children()):
                for layer in list(unit.children()):
                    print(layer)
                    target_layers.append(layer)


            if idx == 0:
                for layer in block.children():
                    if not isinstance(layer, nn.ReLU):
                        target_layers.append(layer)
            elif idx == 1:
                for i in self.layers[idx-1]:
                    if self.block == "BasicBlock":
                        units = [block[i].residual1, block[i].residual2]
                    elif self.block == "Bottleneck":
                        units = [block[i].residual1, block[i].residual2, block[i].residual3]
            else:

                units = [block.]
            for i in range(len(self.layers)):



    def forward(self, inputs):
        outputs = self.conv_bn_relu(inputs)
        outputs = self.max_pool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)

        return outputs
