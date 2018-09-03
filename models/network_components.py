import torch.nn as nn


class Conv2dRelu(nn.Module):
    def __init__(self, in_channels, out_channels, k_size,
                 stride=1, padding=1, dilation=1, bias=True):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)
        return outputs


class Conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, k_size,
                 stride=1, padding=1, dilation=1, bias=True):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        return outputs


class Conv2dBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, k_size,
                 stride=1, padding=1, dilation=1, bias=True):
        super(Conv2dBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock Module
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, down_sample=False):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.residual1 = Conv2dBatchNormRelu(in_channels, out_channels, 3,
                                                 stride=2, bias=False)
            self.residual2 = Conv2dBatchNorm(out_channels, out_channels, 3, bias=False)
            self.shortcut = Conv2dBatchNorm(in_channels, out_channels, 1,
                                            stride=2, bias=False)
        else:
            self.residual1 = Conv2dBatchNormRelu(in_channels, out_channels, 3, bias=False)
            self.residual2 = Conv2dBatchNorm(out_channels, out_channels, 3, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.residual1(inputs)
        outputs = self.residual2(outputs)
        if self.down_sample:
            shortcut = self.shortcut(inputs)
            outputs += shortcut
        else:
            outputs += inputs

        outputs = self.relu(outputs)

        return outputs


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck Module
    """
    expansion = 4

    def __init__(self, in_channels, aux_channels, down_sample=False):
        super(Bottleneck, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.residual1 = Conv2dBatchNormRelu(in_channels, aux_channels, 1,
                                                 stride=2, bias=False)
            self.residual2 = Conv2dBatchNormRelu(aux_channels, aux_channels, 3, bias=False)
            self.residual3 = Conv2dBatchNorm(aux_channels,
                                             aux_channels*self.expansion,
                                             1, bias=False)
            self.shortcut = Conv2dBatchNorm(in_channels, aux_channels*self.expansion,
                                            1, stride=2, bias=False)
        else:
            self.residual1 = Conv2dBatchNormRelu(in_channels, aux_channels, 1, bias=False)
            self.residual2 = Conv2dBatchNormRelu(aux_channels, aux_channels, 3, bias=False)
            self.residual3 = Conv2dBatchNorm(aux_channels,
                                             aux_channels*self.expansion,
                                             1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.residual1(inputs)
        outputs = self.residual2(outputs)
        outputs = self.residual3(outputs)
        if self.down_sample:
            shortcut = self.shortcut(inputs)
            outputs += shortcut
        else:
            outputs += inputs

        outputs = self.relu(outputs)

        return outputs
