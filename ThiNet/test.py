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
    # kernel_weight = torch.arange(outch*inch*3*3,dtype=torch.float32).reshape(outch,inch,3,3)
    kernel_weight = torch.randn(outch, inch,3,3)
    # input = torch.arange(inch*5*5,dtype=torch.float32).reshape(1,inch,5,5)
    input = torch.randn(1,inch,5,5)
    # one = torch.ones(1,1,5,5)
    # input = torch.cat([one*i for i in range(1,1+inch)],1)
    print(input)
    conv1 = nn.Conv2d(inch,outch, 3,padding=1,bias=False)
    conv2 = nn.Conv2d(inch,outch*inch, 3,groups = inch,padding=1,bias=False)
    conv1.weight.data = kernel_weight
    group_kernel_index = []
    for i in range(inch):
        group_kernel_index.extend([i+j*inch for j in range(outch)])
    
    group_out_reindex = []
    for i in range(outch):
        group_out_reindex.extend([i+j*outch for j in range(inch)])
    conv2.weight.data = kernel_weight.reshape(inch*outch,1,3,3)[group_kernel_index]
    print(conv2.weight.data)

    out1 = conv1(input)
    out2 = conv2(input)
    out2 = out2[:,group_out_reindex,...]

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

'''
four scenarios for hook
'''
def test_hook_1():
    '''
    scenario 1: 1 input, 1, output(no concat)
    '''
    succeeding_strategy = unet_succeeding_strategy(4)
    preceding_strategy = get_preceding_from_succeding(succeeding_strategy)
    # print(succeeding_strategy)
    # print(preceding_strategy)
    model = UNet(3,3)
    all_layers = get_layers(model)

    # choose the input conv and the succeeding conv to test
    layer_idx = 0
    layer = all_layers[layer_idx]
    inch = layer.in_channels

    inHook = hookYandX(layer)

    c, h, w = (3, 32, 32)
    # input = torch.arange(c*h*w,dtype=torch.float32).reshape(1,c,h,w).cuda()
    input = torch.randn(1,c,h,w).cuda()

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        model(input)
    y = inHook.y
    x = inHook.x
    rawout = inHook.rawout

    idx = 1
    f1 = y[0][:,idx,...]
    f2 = torch.sum(x[0][:,idx*inch:(idx+1)*inch,...], 1, keepdim=True)
    absdiff = torch.abs(f1-f2)
    print(rawout)
    print(f1)
    print(f2)
    print(absdiff)
    print(torch.max(absdiff))

def test_hook_2():
    '''
    scenario 2: 1 concated input(get the former),  output
    '''
    succeeding_strategy = unet_succeeding_strategy(4)
    preceding_strategy = get_preceding_from_succeding(succeeding_strategy)
    model = UNet(3,3)
    all_layers = get_layers(model)

    # choose the input conv and the succeeding conv to test
    layer_idx = 1
    layer = all_layers[layer_idx]

    sl = succeeding_strategy[layer_idx][1]

    precedding_layers = preceding_strategy[sl]

    C = layer.out_channels
    activation_kernel = list(range(C))

    # this snippet can handle concat more than 2 inputs, not only 2
    if len(precedding_layers) > 1:
        kernel_before = 0
        for pl in precedding_layers:
            if layer_idx > pl:
                kernel_before += all_layers[pl].weight.shape[0]
            else:
                break
        activation_kernel = list(range(kernel_before, kernel_before+C))
    
    print(kernel_before)
    print(activation_kernel)

    inch = layer.in_channels

    inHook = hookYandX(layer,activation_kernel=activation_kernel)

    c, h, w = (3, 32, 32)
    # input = torch.arange(c*h*w,dtype=torch.float32).reshape(1,c,h,w).cuda()
    input = torch.randn(1,c,h,w).cuda()

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        model(input)
    y = inHook.y
    x = inHook.x

    idx = 2
    f1 = y[0][:,idx,...]
    f2 = torch.sum(x[0][:,idx*inch:(idx+1)*inch,...], 1, keepdim=True)
    absdiff = torch.abs(f1-f2)
    print(f1)
    print(f2)
    print(absdiff)
    print(torch.max(absdiff))

def test_hook_3():
    '''
    scenario 3: 1 concated input(get the latter),  output
    '''
    succeeding_strategy = unet_succeeding_strategy(4)
    preceding_strategy = get_preceding_from_succeding(succeeding_strategy)
    model = UNet(3,3)
    all_layers = get_layers(model)

    # choose the input conv and the succeeding conv to test
    layer_idx = 15
    layer = all_layers[layer_idx]

    sl = succeeding_strategy[layer_idx][0]

    precedding_layers = preceding_strategy[sl]

    C = layer.out_channels
    activation_kernel = list(range(C))

    # this snippet can handle concat more than 2 inputs, not only 2
    if len(precedding_layers) > 1:
        kernel_before = 0
        for pl in precedding_layers:
            if layer_idx > pl:
                kernel_before += all_layers[pl].weight.shape[0]
            else:
                break
        activation_kernel = list(range(kernel_before, kernel_before+C))
    
    print(kernel_before)
    print(activation_kernel)

    inch = layer.in_channels

    inHook = hookYandX(layer,activation_kernel=activation_kernel)

    c, h, w = (3, 32, 32)
    # input = torch.arange(c*h*w,dtype=torch.float32).reshape(1,c,h,w).cuda()
    input = torch.randn(1,c,h,w).cuda()

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        model(input)
    y = inHook.y
    x = inHook.x

    idx = 1
    f1 = y[0][:,idx,...]
    f2 = torch.sum(x[0][:,idx*inch:(idx+1)*inch,...], 1, keepdim=True)
    absdiff = torch.abs(f1-f2)
    print(f1)
    print(f2)
    print(absdiff)
    print(torch.max(absdiff))


print("end")

if __name__ == "__main__":
    # test_get_layers()
    # test_weight_assign()
    # test_hook_1()
    # test_hook_2()
    test_hook_3()