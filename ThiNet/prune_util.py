import torch
import torch.nn as nn

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
