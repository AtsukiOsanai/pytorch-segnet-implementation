import torch.nn as nn
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
    # TODO
    - Adding dilated resnet feature extractor.
      This causes the model transfer operation from caffe model.
    """


    def __init__(self, resnet_type):
        super(ResNet, self).__init__()
        self.block = resnet_config[resnet_type]["block"]
        self.layers = resnet_config[resnet_type]["layers"]
        self.model_url = resnet_config[resnet_type]["model_url"]

        self.conv_bn_relu = Conv2dBatchNormRelu(3, 64, 7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # TODO Fix the channel mismatch
        self.in_channels = 64
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], down_sample=True)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], down_sample=True)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], down_sample=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, aux_channels, num_layers, down_sample=False):
        layers = []
        layers.append(block(self.in_channels, aux_channels, down_sample))
        for i in range(1, num_layers):
            layers.append(block(aux_channels * block.expansion, aux_channels))
        self.in_channels = aux_channels * block.expansion

        return nn.Sequential(*layers)

    def init_pretrained(self):
        pretrained_params = model_zoo.load_url(self.model_url)
        # extract trainable parameter keys from pre-trained model
        param_keys, conv_keys, bn_keys = [], [], []
        bn_idx = 0
        for key in list(pretrained_params.keys()):
            if "conv" in key or "downsample.0" in key:
                conv_keys.append(key)
                param_keys.append(conv_keys)
                conv_keys = []
            elif "bn" in key or "downsample.1" in key:
                bn_keys.append(key)
                bn_idx += 1
                if bn_idx == 4:
                    param_keys.append(bn_keys)
                    bn_idx = 0
                    bn_keys = []

        # extract trainable layers from defined model
        blocks = [self.conv_bn_relu,
                  self.layer1,
                  self.layer2,
                  self.layer3,
                  self.layer4]
        target_layers = []
        for idx, block in enumerate(blocks):
            if idx < 1:
                for layer in block.children():
                    if not isinstance(layer, nn.ReLU):
                        target_layers.append(layer)
            else:
                for unit in list(block.children()):
                    for layers in list(unit.children()):
                        for layer in layers.children():
                            if not isinstance(layer, nn.ReLU) and \
                               not isinstance(layer, nn.AvgPool2d):
                                target_layers.append(layer)
        # assign the pre-trained weights into the defined model
        for layer, key in zip(target_layers, param_keys):
            if len(key) == 1:  # conv
                layer.weight.data = pretrained_params[key[0]]
            else:  # bn
                layer.running_mean.data = pretrained_params[key[0]]
                layer.running_var.data = pretrained_params[key[1]]
                layer.weight.data = pretrained_params[key[2]]
                layer.bias.data = pretrained_params[key[3]]

    def forward(self, inputs, output_aux=False):
        outputs = self.conv_bn_relu(inputs)
        outputs = self.max_pool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        aux = self.layer3(outputs)
        outputs = self.layer4(aux)
        outputs = self.avgpool(outputs)

        if outputs_aux:
            return aux, outputs
        else:
            return outputs
