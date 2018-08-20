import torcn.nn as nn


class Conv2dBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, k_size,
                 padding=1, stride=1, bias=True, dilation=1):
        super(Conv2dBatchNormRelu, self).__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, k_size,
                           padding=padding, stride=stride, bias=bias, dilation=dilation)
        self.unit = nn.Sequential(conv2d,
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),  # inplace=True
                                  )

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs
