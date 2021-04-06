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

    inch = layer.out_channels

    slayer = all_layers[sl]

    inHook = hookYandX(slayer,activation_kernel=activation_kernel)

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

    inch = layer.out_channels

    slayer = all_layers[sl]

    inHook = hookYandX(slayer,activation_kernel=activation_kernel)

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

def test_hook_4():
    model = UNet(3,3)
    layer_idx = 16
    all_layers = get_layers(model)
    layer = all_layers[layer_idx]

    activation_kernel = list(range(layer.in_channels))

    af = activation_kernel[:64]
    al = activation_kernel[64:]

    formerHook = hookYandX(layer, af)
    latterHook = hookYandX(layer, al)

    c, h, w = (3, 32, 32)
    # input = torch.arange(c*h*w,dtype=torch.float32).reshape(1,c,h,w).cuda()
    input = torch.randn(1,c,h,w).cuda()

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        model(input)
    
    yf = formerHook.y[0]
    yl = latterHook.y[0]

    out = yf+yl

    rawout = formerHook.rawout

    absdiff = torch.abs(rawout - out)
    print(absdiff)
    print(torch.max(absdiff))

def test_collecting_reshape():
    batch, inch, outch = (2,3,4)
    h, w = (5,5)
    group_out = torch.randn(batch,inch*outch,h,w)
    outs = []
    for i in range(outch):
        outs.append(torch.sum(group_out[:,i*inch:(i+1)*inch,...],1, keepdim=True))
    out = torch.cat(outs,1)
    out = out.view(-1,1)
    sample_num = out.shape[0]
    group_out = group_out.contiguous().view(-1, outch, inch, h, w)
    group_out = group_out.permute(0, 1, 3, 4, 2)
    group_out = group_out.contiguous().view(-1, inch)
    print(out)
    print(group_out)
    absdiff = torch.abs(out- torch.sum(group_out,1,keepdim=True))
    print(absdiff)
    print(torch.max(absdiff))

def test_collecting_training_examples():
    model = UNet(3,3).cuda()
    layer_idx = 5
    all_layers = get_layers(model)
    layer = all_layers[layer_idx]
    train_loader = []
    train_sample_num = 2
    for i in range(train_sample_num):
        train_loader.append({'L':torch.randn(1,3,32,32)})
    x, y , sample_num = collecting_training_examples(model, layer,train_loader)
    print("y",y)
    print("x",x)
    absdiff = torch.abs(y- torch.sum(x,1,keepdim=True))
    print("absdiff",absdiff)
    print(torch.max(absdiff))

def test_get_subset():
    pass

def test_LSE():
    m, n = 10,3
    A = torch.randn(m,n)
    B = torch.randn(m,1)
    res1 = torch.lstsq(B, A)[0][:n]
    print(res1)
    res2 = get_w_by_LSE(A, B)
    print(res1.shape)
    print(res2.shape)
    print("res1",res1)
    print("res2",res2)
    absdiff = torch.abs(res1-res2)
    print("absdiff",absdiff)
    print(torch.max(absdiff))

def test_LSE_when_not_full_rank():
    m, n = 100,30
    A = torch.randn(m,n)
    B = torch.randn(m,1)
    
    A[:,3] = A[:,4]+A[:,5]
    A[:,0] = A[:,7]+A[:,10]+A[:,21]
    print(torch.matrix_rank(A))
    w, mlis = get_w_by_LSE(A, B)
    print(w)
    print(mlis)
    diff = B - torch.mm(A[:,mlis],w)
    print(diff**2)
    print(torch.max(diff**2))
    
def test_thinet_prune(layer_idx):
    model = UNet(3,3).cuda()
    train_loader = []
    train_sample_num = 2
    r = 13/16
    for i in range(train_sample_num):
        train_loader.append({'L':torch.randn(2,3,32,32)})
    newmodel = thinet_prune_layer(model, layer_idx, train_loader, r)
    print(newmodel is model)

    # test if the pruned model can properly inference.
    input = torch.randn(2,3,32,32).cuda()
    model(input)
    print("inference ok")

    for n, p in model.named_parameters():
        print(n, p.shape)


print("end")

if __name__ == "__main__":
    # test_get_layers()
    # test_weight_assign()
    # test_hook_1()
    # test_hook_2()
    # test_hook_3()
    # test_hook_4()
    # test_collecting_reshape()
    # test_collecting_training_examples()
    # test_LSE()
    # test_LSE_when_not_full_rank()
    test_thinet_prune(1)
    test_thinet_prune(15)