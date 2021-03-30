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


def get_w_by_LSE():
    pass

def get_layers(model):
    pass

def thinet_prune_layer(model, layer, train_loader, r, m=1000):
    C = layer.out_channel
    x, y = collecting_training_examples(model, layer, train_loader, m)
    subset = get_subset(x, y, r, C)
    assert len(set(subset)) == len(subset) # assure no duplicate element
    
    '''
    TODO:prune the layer
    '''


    return model

def save_model()
    pass
