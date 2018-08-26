import torch.nn as nn
from network_components import *
from torchvision import models


class segnet(nn.Module):

    def __init__(self, n_classes=21, in_channels=3):
        super(segnet, self).__init__()

        # Encoder
        self.encoder1 = encoder_conv2(in_channels, 64)
        self.encoder2 = encoder_conv2(64, 128)
        self.encoder3 = encoder_conv3(128, 256)
        self.encoder4 = encoder_conv3(256, 512)
        self.encoder5 = encoder_conv3(512, 512)

        # Decoder
        self.decoder5 = decoder_conv3(512, 512)
        self.decoder4 = decoder_conv3(512, 256)
        self.decoder3 = decoder_conv3(256, 128)
        self.decoder2 = decoder_conv2(128, 64)
        self.decoder1 = decoder_last(64, n_classes)

        self.softmax = nn.Softmax()

    def forward(self, inputs):
        # Encoder
        outputs, indices_1, unpool_shape1 = self.encoder1(inputs)
        outputs, indices_2, unpool_shape2 = self.encoder2(outputs)
        outputs, indices_3, unpool_shape3 = self.encoder3(outputs)
        outputs, indices_4, unpool_shape4 = self.encoder4(outputs)
        outputs, indices_5, unpool_shape5 = self.encoder5(outputs)

        # Decoder
        outputs = self.decode5(outputs, indices_5, unpool_shape5)
        outputs = self.decode4(outputs, indices_4, unpool_shape4)
        outputs = self.decode3(outputs, indices_3, unpool_shape3)
        outputs = self.decode2(outputs, indices_2, unpool_shape2)
        outputs = self.decode1(outputs, indices_1, unpool_shape1)

        outputs = self.softmax(outputs)

        return outputs

    def init_vgg16bn_params(self):
        vgg16bn = models.vgg16_bn(pretrained=True)
        # extract the vgg16_bn layer without ReLU
        vgg16bn_features = list(vgg16bn.features.children())
        vgg16bn_layers = []
        for layer in features:
            if not isinstance(layer, nn.ReLU):
                vgg16bn_layers.append(layer)

        # extract the segnet encoder layer without ReLU
        target_layers = []
        blocks = [self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4,
                  self.encoder5]
        for idx, block in enumerate(blocks):
            if idx < 2:
                units = [block.conv1.unit, block.conv2.unit]
            else:
                units = [block.conv1.unit, block.conv2.unit, block.conv3.unit]

            for unit in units:
                for layer in unit:
                    if not isinstance(layer, nn.ReLU):
                        target_layers.append(layer)

        assert len(vgg16bn_layers) == len(target_layers), "Found a segnet and pre-trained model size mismatch."
        # adapt the pre-trained weights and biases
        for target_layer, vgg16bn_layer in zip(target_layers, vgg16bn_layers):
            target_layer.weight.data = vgg16bn_layer.weight.data
            target_layer.bias.data = vgg16bn_layer.bias.data


class encoder_conv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_conv2, self).__init__()

        self.conv1 = Conv2dBatchNormRelu(in_channels, out_channels, 3)
        self.conv2 = Conv2dBatchNormRelu(out_channels, out_channels, 3)
        self.maxpool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool(outputs)

        return outputs, indices, unpooled_shape


class encoder_conv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_conv3, self).__init__()

        self.conv1 = Conv2dBatchNormRelu(in_channels, out_channels, 3)
        self.conv2 = Conv2dBatchNormRelu(out_channels, out_channels, 3)
        self.conv3 = Conv2dBatchNormRelu(out_channels, out_channels, 3)
        self.maxpool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool(outputs)

        return outputs, indices, unpooled_shape


class decoder_conv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_conv2, self).__init__()

        self.unpool = nn.MaxUnpool2d(2)
        self.conv1 = Conv2dBatchNormRelu(in_channels, in_channels, 3)
        self.conv2 = Conv2dBatchNormRelu(in_channels, out_channels, 3)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs


class decoder_conv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_conv3, self).__init__()

        self.unpool = nn.MaxUnpool2d(2)
        self.conv1 = Conv2dBatchNormRelu(in_channels, in_channels, 3)
        self.conv2 = Conv2dBatchNormRelu(in_channels, in_channels, 3)
        self.conv3 = Conv2dBatchNormRelu(in_channels, out_channels, 3)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        return outputs


class decoder_last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_last, self).__init__()

        self.unpool = nn.MaxUnpool2d(2)
        self.conv1 = Conv2dBatchNormRelu(in_channels, in_channels, 3)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs
