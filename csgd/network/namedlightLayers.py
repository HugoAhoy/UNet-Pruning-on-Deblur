import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
        #                        bias=bias)
        # self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.sep = nn.Sequential(OrderedDict([
            ('1conv',nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                               bias=bias)),
            ('pointwiseconv',nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias))
        ]))

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.pointwise(x)
        x = self.sep(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels // 4, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels // 4, out_channels, kernel_size=5, padding=2),

        )

    def forward(self, x):
        return x + self.double_conv(x)
        #return self.double_conv(x)


class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)

        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels // 4, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels // 4, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.downsample(x) + self.double_conv(x)
        #return self.double_conv(x)


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.double_conv(x)
        #return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, sep_in_channels):
        super().__init__()
        self.decoder = Decoder(in_channels, in_channels)
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.sep = SeparableConv2d(sep_in_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.decoder(x1)
        # input is CHW
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2 = self.sep(x2)
        return x1+x2



