import torch.nn as nn
import torch.nn.functional as F
# from network.layers import *
from network.namedlayers import *


class UNet(nn.Module):

    def __init__(self, inChannels, outChannels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = inChannels
        self.n_classes = outChannels
        self.bilinear = bilinear

        self.inc = DoubleConv(inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, outChannels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ScaleUNet(nn.Module):

    def __init__(self, inChannels, outChannels, ScaleUp = 1, ScaleDown = 1, bilinear=True):
        super(ScaleUNet, self).__init__()
        self.n_channels = inChannels
        self.n_classes = outChannels
        self.bilinear = bilinear
        self.su = ScaleUp
        self.sd = ScaleDown

        self.inc = DoubleConv(inChannels, 64*self.su // self.sd)
        self.down1 = Down(64*self.su // self.sd, 128*self.su // self.sd)
        self.down2 = Down(128*self.su // self.sd, 256*self.su // self.sd)
        self.down3 = Down(256*self.su // self.sd, 512*self.su // self.sd)
        factor = 2 if bilinear else 1
        self.down4 = Down(512*self.su // self.sd, 1024 // factor*self.su // self.sd)
        self.up1 = Up(1024*self.su // self.sd, 512 // factor*self.su // self.sd, bilinear)
        self.up2 = Up(512*self.su // self.sd, 256 // factor*self.su // self.sd, bilinear)
        self.up3 = Up(256*self.su // self.sd, 128 // factor*self.su // self.sd, bilinear)
        self.up4 = Up(128*self.su // self.sd, 64*self.su // self.sd, bilinear)
        self.outc = OutConv(64*self.su // self.sd, outChannels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    #from torchstat import stat
    #from thop import profile, clever_format
    net = UNet(3,3)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (3, 3000, 4000), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
