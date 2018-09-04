import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from network_components import *


class fcn(nn.Module):
    def __init__(self, n_classes, learned_bilinear=False, pre_trained=False):
        super(fcn, self).__init__()
        self.learned_bilinear = learned_bilinear

        self.conv1 = first_conv2(3, 64)
        self.conv2 = conv2(64, 128)
        self.conv3 = conv3(128, 256)
        self.conv4 = conv3(256, 512)
        self.conv5 = conv3(512, 512)

        self.classifier = nn.Sequential(
            Conv2dRelu(512, 4096, 7, padding=0),
            nn.Dropout2d(),
            Conv2dRelu(4096, 4096, 1, padding=0),
            nn.Dropout2d(),
            nn.Conv2d(4096, n_classes, padding=0)
        )
        self.softmax = nn.Softmax(dim=1)

        if pre_trained:
            self.init_vgg16_params()

    def init_vgg16_params(self):
        vgg16 = models.vgg16(pretrained=True)
        # extract the vgg16 layer without ReLU
        vgg16_features = list(vgg16.features.children())
        vgg16_layers = []
        for layer in vgg16_features:
            if not isinstance(layer, nn.ReLU):
                vgg16_layers.append(layer)

        # extract the fcn encoder layer without ReLU
        target_layers = []
        blocks = [self.conv1,
                  self.conv2,
                  self.conv3
                  self.conv4,
                  self.conv5]
        for idx, block in enumerate(blocks):
            if idx < 2:
                units = [block.conv1.unit, block.conv2.unit]
            else:
                units = [block.conv1.unit, block.conv2.unit, block.conv3.unit]

            for unit in units:
                for layer in unit:
                    if not isinstance(layer, nn.ReLU):
                        target_layers.append(layer)

        assert len(vgg16_layers) == len(target_layers), "Found a size mismatch between fcn and pre-trained model."
        # adapt the pre-trained weights and biases
        for target_layer, vgg16_layer in zip(target_layers, vgg16_layers):
            target_layer.weight.data = vgg16_layer.weight.data
            target_layer.bias.data = vgg16_layer.bias.data



class fcn32s(fcn):
    def __init__(self, n_classes, learned_bilinear=False, pre_trained=False):
        super(fcn32s, self).__init__(n_classes, learned_bilinear, pre_trained)

        if learned_bilinear:
            self.upsample = nn.ConvTranspose2d(n_classes, n_classes,
                                               64, stride=32,
                                               bias=False)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        scores = self.classifier(conv5)
        if self.learned_bilinear:
            scores = self.upsample(scores)
        else:
            scores = nn.functional.interpolate(
                scores, size=inputs.size()[2:], mode='bilinear')

        scores = self.softmax(scores)

        return scores


class fcn16s(fcn):
    def __init__(self, n_classes, learned_bilinear=False, pre_trained=False):
        super(fcn16s, self).__init__(n_classes, learned_bilinear, pre_trained)

        if learned_bilinear:
            self.upsample = nn.ConvTranspose2d(n_classes, n_classes,
                                               4, stride=2,
                                               bias=False)
            self.upsample2 = nn.ConvTranspose2d(n_classes, n_classes,
                                                32, stride=16,
                                                bias=False)

        self.score_pool4 = nn.Conv2d(512, n_classes, 1, bias=False)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        scores = self.classifier(conv5)

        scores_pool4 = self.score_pool4(conv4)

        if self.learned_bilinear:
            scores = self.upsample(scores)
            scores += scores_pool4
            scores = self.upsample2(scores)
        else:
            scores = nn.functional.interpolate(
                scores, size=scores_pool4.size()[2:], mode='bilinear')
            scores += upsample_pool4
            scores = nn.functional.interpolate(
                scores, size=inputs.size()[2:], mode='bilinear')

        scores = self.softmax(scores)

        return scores


class fcn8s(fcn):
    def __init__(self, n_classes, learned_bilinear=False, pre_trained=False):
        super(fcn8s, self).__init__(n_classes, learned_bilinear, pre_trained)

        if learned_bilinear:
            self.upsample = nn.ConvTranspose2d(n_classes, n_classes,
                                               4, stride=2,
                                               bias=False)
            self.upsample2 = nn.ConvTranspose2d(n_classes, n_classes,
                                                4, stride=2,
                                                bias=False)
            self.upsample3 = nn.ConvTranspose2d(n_classes, n_classes,
                                                16, stride=8,
                                                bias=False)

        self.score_pool3 = nn.Conv2d(256, n_classes, 1, bias=False)
        self.score_pool4 = nn.Conv2d(512, n_classes, 1, bias=False)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        scores = self.classifier(conv5)

        scores_pool3 = self.score_pool3(conv3)
        scores_pool4 = self.score_pool4(conv4)

        if self.learned_bilinear:
            scores = self.upsample(scores)
            scores += scores_pool4
            scores = self.upsample2(scores)
            scores += scores_pool3
            scores = self.upsample3(scores)
        else:
            scores = nn.functional.interpolate(
                scores, size=scores_pool4.size()[2:], mode='bilinear')
            scores += scores_pool4
            scores = nn.functional.interpolate(
                scores, size=scores_pool3.size()[2:], mode='bilinear')
            scores += scores_pool3
            scores = nn.functional.interpolate(
                scores, size=inputs.size()[2:], mode='bilinear')

        scores = self.softmax(scores)

        return scores


class first_conv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(first_conv2, self).__init__()

        self.conv1 = Conv2dRelu(in_channels, out_channels, 3, padding=100)
        self.conv2 = Conv2dRelu(out_channels, out_channels, 3)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.maxpool(outputs)
        return outputs


class conv2(first_conv2):
    def __init__(self, in_channels, out_channels):
        super(conv2, self).__init__(in_channels, out_channels)

        self.conv1 = Conv2dRelu(in_channels, out_channels, 3)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.maxpool(outputs)
        return outputs


class conv3(conv2):
    def __init__(self, in_channels,  out_channels):
        super(conv3, self).__init__(in_channels, out_channels)

        self.conv3 = Conv2dRelu(out_channels, out_channels, 3)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.maxpool(outputs)
        return outputs
