import torch.nn as nn
import torch.nn.functional as F
# from network.layers import *
from network.namedlayers import *
from prune_util import parse_filternum_dict

class ArbitaryUNet(nn.Module):

    def __init__(self, inChannels, outChannels, filternum_dict, bilinear=True):
        super(UNet, self).__init__()
        self.stage = ((len(filternum_dict)-1)//2 - 1)//2
        self.n_channels = inChannels
        self.n_classes = outChannels
        self.bilinear = bilinear
        settings = parse_filternum_dict(filternum_dict)

        incsetting = settings['inc']
        self.inc = DoubleConv(incsetting.inch, incsetting.out, incsetting.mid)

        for i in range(1, self.stage+1):
            downsetting = settings['down{}'.format(i)]
            setattr(self,'down{}'.format(i), Down(downsetting.inch, downsetting.out, downsetting.mid))

        for i in range(1, self.stage+1):
            upsetting = settings['up{}'.format(i)]
            setattr(self,'up{}'.format(i), Up(downsetting.inch, downsetting.out, mid_channels=downsetting.mid))

        outsetting = settings['out']
        self.outc = OutConv(outsetting.inch, outsetting.out)

    def forward(self, x):
        outs = []
        out['x0'] = self.inc(x)
        for i in range(1, self.stage+1):
            out['x{}'.format(i)] = getattr(self, 'down{}'.format(i))(out['x{}'.format(i-1)])
        
        x = out['x{}'.format(self.stage)]
        for i in range(self.stage):
            x = getattr(self, 'up{}'.format(i))(x, out['x{}'.format(stage-i)])

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
