import torch.nn as nn

from network_components import *


class unet(nn.Module):
    def __init__(self, n_classes):
        super(unet, self).__init__()

        self.unet_down1 = unet_down(3, 64)
        self.unet_down2 = unet_down(64, 128)
        self.unet_down3 = unet_down(128, 256)
        self.unet_down4 = unet_down(256, 512)
        self.unet_bottom = unet_bottom(512, 1024)
        self.unet_up4 = unet_up(1024, 512)
        self.unet_up3 = unet_up(512, 256)
        self.unet_up2 = unet_up(256, 128)
        self.unet_up1 = unet_up(128, 64)
        self.last_layer = nn.Conv2d(64, n_classes, 3, stride=1, padding=1)

    def forward(self, inputs):
        features1 = self.unet_down1(inputs)
        features2 = self.unet_down2(features1)
        features3 = self.unet_down3(features2)
        features4 = self.unet_down4(features3)

        outputs = self.unet_bottom(features4)

        outputs = self.unet_up4(outputs, features4)
        outputs = self.unet_up3(outputs, features3)
        outputs = self.unet_up2(outputs, features2)
        outputs = self.unet_up1(outputs, features1)

        outupts = self.last_layer(outputs)

        return outputs


class unet_down(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(unet_down, self).__init__()

        self.conv_bn_relu1 = Conv2dBatchNormRelu(in_channels, out_channels, k_size)
        self.conv_bn_relu2 = Conv2dBatchNormRelu(out_channels, out_channels, k_size)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        outputs = self.conv_bn_relu1(inputs)
        outputs = self.conv_bn_relu2(outputs)
        outputs = self.max_pool(outputs)

        return outputs


class unet_bottom(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(unet_bottom, self).__init__()

        self.conv_bn_relu1 = Conv2dBatchNormRelu(in_channels, out_channels, k_size)
        self.conv_bn_relu2 = Conv2dBatchNormRelu(out_channels, out_channels, k_size)

    def forward(self, inputs):
        outputs = self.conv_bn_relu1(inputs)
        outputs = self.conv_bn_relu2(inputs)

        return outputs


class unet_up(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(unet_up, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu1 = Conv2dBatchNormRelu(2 * out_channels, out_channels, k_size)
        self.conv_bn_relu2 = Conv2dBatchNormRelu(out_channels, out_channels, k_size)

    def forward(self, inputs, skips):
        outputs = self.up(inputs)
        outputs = self.conv_bn_relu1(torch.cat([outputs, skips], dim=1))
        outputs = self.conv_bn_relu2(outputs)

        return outputs
