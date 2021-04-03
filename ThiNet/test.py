from network.UNet import UNet
from thinet_utils import *
from prune_util import *
import torch

def test_get_layers():
    model = UNet(3, 3)
    all_layers = get_layers(model)
    for l in all_layers:
        print(l)
        print("-"*30)


print("end")

if __name__ == "__main__":
    test_get_layers()