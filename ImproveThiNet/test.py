from network.UNet import UNet
from improve_thinet_utils import *
from prune_util import *
import torch

def test_pipeline():
    model = UNet(3,3).cuda()
    train_loader = []
    train_sample_num = 2
    r = 13/16
    for i in range(train_sample_num):
        train_loader.append({'L':torch.randn(2,3,32,32)})
    pruned_filter_dict, pruned_filter_dict_thinner = improve_thinet_pruned_structure(model, train_loader, r, gpu_id=0, min_channel_ratio=0.5, decay_factor=0.96, m=4)
    print("pruned_filter_dict:",pruned_filter_dict)
    print("pruned_filter_dict_thinner:",pruned_filter_dict_thinner)

if __name__ == "__main__":
    test_pipeline()