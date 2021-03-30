import torch
import numpy as np
import torch.nn as nn
import random

'''
for get coresponding input and output
'''
class hookYandX:
    self.y = []
    self.x = []
    def __init__(self, layer):
        self.layer = layer
        inch, outch, k, p, s = layer.in_channel, layer.out_channel, layer.kernel_size, layer.padding, layer.stride
        self.channel_wise = nn.Conv2d(inch, outch*inch,k, groups=inch, padding=p, stride=s, bias=False)
        self.channel_wise.weight.data = self.layer.weight.data.reshape(outch*inch, 1, k[0],k[1])
        self.hook = layer.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        if layer.bias is not None:
            output = output-layer.bias
        self.y.append(output.detach())
        self.channel_wise = self.channel_wise.cuda()
        self.x.append(self.channel_wise(input).detach())

    def remove(self):
        self.hook.remove()

'''
collecting training examples for layer pruning
'''
def collecting_training_examples(model, layer, train_loader, m=1000):
    in_and_out = hookYandX(layer)
    inch = layer.in_channel
    model.eval()
    for i, train_data in enumerate(train_loader):
        with torch.no_grad():
            train_data['L'] = train_data['L'].cuda()
            model(train_data['L'])
    
    y = tf.concat(in_and_out.y,0).contiguous().view(-1,1)
    samplesize = y.shape[0]

    m = min(m, samplesize)
    selected_index = random.sample(range(samplesize), m)
    x = tf.concat(in_and_out.x, 0).contiguous().view(samplesize, inch)

    y = y[selected_index,:]
    x = x[selected_index,:]
    return(x, y)


def get_subset(x, y, r, C):
    '''
    x:shape(sample_size, C)
    y:shape(sample_size, 1)
    r is the compression ratio as mentioned in paper;
    C is the channel num of the filter
    '''

    T = int(C*(1-r))
    res = []

    # greedy
    for iter in range(T):
        min_value = float('inf')
        tempRes = None
        for i in range(C):
            if i in res:
                continue
            else:
                tempT = res + [i]
                mask = torch.sum(torch.eye(C)[tempT,:], dim=1, keepdim=True) # mask shape(iter, C)
                diff = y-torch.sum(mask*x, dim=1,keepdim=True)
                total_error = torch.sum(diff**2)
                if total_error < min_value:
                    min_value = total_error
                    tempRes = tempT
        res = tempRes
    return res


def get_w_by_LSE(x, y):
    a = torch.matmul(torch.transpose(x,0,1),x)
    if torch.matrix_rank(a) == a.shape[0]:
        a_inv = torch.inverse(a)
    else:
        a_inv = torch.pinverse(a)
    w = torch.chain_matmul(a_inv, torch.transpose(x, 0,1), y)
    return w

def get_layers(model):
    pass

def get_succeeding_layers(layer):
    pass

def thinet_prune_layer(model, layer, train_loader, r, m=1000):
    C = layer.out_channel
    x, y = collecting_training_examples(model, layer, train_loader, m)
    prune_subset = get_subset(x, y, r, C)
    assert len(set(prune_subset)) == len(prune_subset) # assure no duplicate element


    '''
    prune the layer
    '''
    saved_subset = list(set(range(C))-set(prune_subset))
    w = get_w_by_LSE(x[:,saved_subset], y)

    layer.weight.data = layer.weight.data[saved_subset, ...] 
    assert layer.weight.data.shape[1] == w.shape[0] # filter num should be the same as the element num of w
    layer.weight.data = layer.weight.data * (w.unsqueeze(0).unsqueeze(-1).expand(layer.weight.shape))
    layer.weight.grad = None

    if layer.bias is not None:
        layer.bias.data = layer.bias.data[saved_subset,...]
        layer.bias.grad = None
    
    '''
    prune the succeeding layer
    '''
    succeeding_layers = get_succeeding_layers(layer)
    for sl in succeeding_layers:
        sl.weight.data = sl.weight.data[:,saved_subset,...]
        sl.weight.grad = None
        if sl.bias is not None:
            sl.bias.data = sl.bias.data[saved_subset,...]
            sl.bias.grad = None

    return model

def save_model()
    pass
