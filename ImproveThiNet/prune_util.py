import torch
import torch.nn as nn
from collections import namedtuple

def get_filter_num_by_layer(model):
    filter_num = []
    for i in model.modules():
        if isinstance(i, nn.Conv2d):
            filter_num.append(i.weight.shape[0])
    return filter_num

def get_preceding_from_succeding(succeeding_map):
    preceding_map = {}
    for k, v in succeeding_map.items():
        if type(v) is not list:
            v = [v]
        for i in v:
            if i not in preceding_map:
                preceding_map[i] = [k]
            else:
                preceding_map[i].append(k)
    for k in preceding_map:
        preceding_map[k].sort()
    return preceding_map

def unet_succeeding_strategy(n):
    assert n > 0
    result = {}
    totalconv = (n*2+1)*2+1
    for i in range(totalconv - 1):
        if i < 2*4 and i%2 == 1:
            result[i] = [i+1, totalconv - 2 - i]
        else:
            result[i] = [i+1]
    result[totalconv-1] = []
    return result

if __name__ == "__main__":
    from network.UNet import UNet
    model = UNet(3,3)
    suc = unet_succeeding_strategy(4)
    pre = get_preceding_from_succeding(suc)
    filter_num = get_filter_num_by_layer(model)
    print(suc)
    print(pre)
    print(filter_num)
    print(len(suc), len(pre), len(filter_num))

'''
Return the setting of inconv, down, up, outconv
all stuff are in namedtuple
'''

def parse_filternum_dict(filternum_dict):
    stage = ((len(filternum_dict) -1) //2-1)//2
    succeeding_strategy = unet_succeeding_strategy(stage)
    preceding_strategy = get_preceding_from_succeding(preceding_strategy)

    DoubleConvSetting = namedtuple('DoubleConvSetting', ['inch','out', 'mid'])
    ConvSetting = namedtuple('ConvSetting',['inch','out'])
    settingdict = {}

    inc_in = 3
    inc_mid = filternum_dict[0]
    inc_out = filternum_dict[1]
    settingdict['inc'] = DoubleConvSetting(inc_in, inc_out, inc_mid)
    for i in range(stage):
        inch = 0
        for j in preceding_strategy[2+i*2]:
            inch += filternum_dict[j]

        mid = 0
        for j in preceding_strategy[2+i*2]:
            mid += filternum_dict[j]

        out = filternum_dict[2+i*2+1]
        settingdict['down{}'.format(i)] = DoubleConvSetting(inch, mid, out)

    for i in range(stage, stage*2):
        inch = 0
        for j in preceding_strategy[2+i*2]:
            inch += filternum_dict[j]

        mid = 0
        for j in preceding_strategy[2+i*2]:
            mid += filternum_dict[j]

        out = filternum_dict[2+i*2+1]
        settingdict['up{}'.format(i-stage)] = DoubleConvSetting(inch, mid, out)
    
    outconv_in = filternum_dict(len(filternum_dict)-1)
    settingdict['out'] = ConvSetting(outconv_in, 3)

    return settingdict