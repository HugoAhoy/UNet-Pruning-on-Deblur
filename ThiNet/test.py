from network.UNet import UNet
from thinet_utils import *
from prune_util import *
import torch
import torch.nn

def test_get_layers():
    model = UNet(3, 3)
    all_layers = get_layers(model)
    for l in all_layers:
        print(l)
        print("-"*30)


def test_weight_assign():
    inch, outch = 3,2
    kernel_weight = torch.arange(outch*inch*3*3,dtype=torch.float32).reshape(outch,inch,3,3)
    # kernel_weight = torch.randn(outch, inch,3,3)
    input = torch.arange(inch*5*5,dtype=torch.float32).reshape(1,inch,5,5)
    # input = torch.randn(1,inch,5,5)
    print(input)
    conv1 = nn.Conv2d(inch,outch, 3,padding=1,bias=False)
    conv2 = nn.Conv2d(inch,outch*inch, 3,groups = inch,padding=1,bias=False)
    conv1.weight.data = kernel_weight
    conv2.weight.data = kernel_weight.reshape(inch*outch,1,3,3)
    
    input = input
    conv1 = conv1
    conv2 = conv2

    out1 = conv1(input)
    out2 = conv2(input)

    print("out1",out1)
    print("out2",out2)

    out2sum = []
    for i in range(outch): 
        out2sum.append(torch.sum(out2[:,i*inch:(i+1)*inch,...], 1,keepdim=True))
    out2sum = torch.cat(out2sum,1)
    print(out2sum.shape)
    diff = out1-out2sum
    absdiff = torch.abs(diff)
    print(absdiff)
    print(torch.max(diff))


print("end")

if __name__ == "__main__":
    # test_get_layers()
    test_weight_assign()