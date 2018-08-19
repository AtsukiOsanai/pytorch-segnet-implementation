import torcn.nn as nn


class Conv2dBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, k_size,
                 padding=1, stride=1, bias=True, dilation=1):
        super(Conv2dBatchNormRelu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, k_size,
            padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.conv2d(inputs)
        outputs = nn.BatchNorm2d(outputs)
        outputs = nn.ReLU(outputs)

        return outputs
