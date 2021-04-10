import torch
import numpy as np
import torch.nn as nn
import random
from prune_util import get_preceding_from_succeding, get_filter_num_by_layer
from prune_util import unet_succeeding_strategy
from itertools import combinations
import math

'''
for get coresponding input and output
'''
class hookYandX:
    def __init__(self, layer,activation_kernel=None):
        # self.y = []
        self.x = []
        self.input = []
        self.rawout = None # this is for test
        self.activation_kernel = activation_kernel
        self.layer = layer
        if activation_kernel is None:
            self.activation_kernel = list(range(layer.weight.shape[1]))
        self.inch = len(self.activation_kernel)
        self.outch, self.k, p, s = layer.out_channels, layer.kernel_size, layer.padding, layer.stride

        # self.conv = nn.Conv2d(inch, outch,k, groups=1, padding=p, stride=s, bias=False)
        self.channel_wise = nn.Conv2d(self.inch, self.inch,self.k, groups=self.inch, padding=p, stride=s, bias=False)
        self.trimmed_weight = layer.weight.data[:,self.activation_kernel,...]
        # for test begin
        # kernel = torch.arange(torch.numel(layer.weight), dtype=torch.float32).reshape(layer.weight.shape)
        # trimmed_weight = kernel[:,self.activation_kernel,...]
        # for test end
        # self.conv.weight.data = trimmed_weight
    
    def hook_fn(self, module, input, output):
        input = input[0]
        input = input[:,self.activation_kernel,...].detach().cpu()
        self.input.append(input)
    
    def get_x(self, gpu_id):
        '''
        .cuda(input.device) make sure the conv weight and input are on same cuda device
        '''
        f = random.sample(range(self.outch), len(self.input)) # choose one filter each batch
        for i in range(len(self.input)):
            device = gpu_id
            input = self.input[-1].cuda(device)
            self.input.pop()
            
            self.channel_wise.weight.data = self.trimmed_weight[f[i]].reshape(self.inch, 1, self.k[0],self.k[1])
            self.channel_wise = self.channel_wise.to(device).eval()
            with torch.no_grad():
                group_output = self.channel_wise(input)
                group_output = group_output.detach().cpu()

            '''
            each input get one sample point
            '''
            out = []
            sample_num = group_output.shape[0]
            '''
            when the feature map is small, the sample num may be bigger then h and w,
            so if the h or w < sample num, I generate the duplicate range, to enable the sampling.
            '''
            duph = math.ceil(sample_num/group_output.shape[2])
            dupw = math.ceil(sample_num/group_output.shape[3])
            hrange = range(group_output.shape[2])
            wrange = range(group_output.shape[3])
            if duph > 1:
                hrange = list(hrange)*duph
            if dupw > 1:
                wrange = list(wrange)*dupw
            
            h = random.sample(hrange, sample_num)
            w = random.sample(wrange, sample_num)
            for k in range(sample_num):
                out.append(group_output[k, :,h[i],w[i]].reshape(1,self.inch))

            self.x.extend(out)

    def register(self):
        self.hook = self.layer.register_forward_hook(self.hook_fn)

    def remove(self):
        self.hook.remove()

class FilterChannelSelection:
    def __init__(self, x, y, decay_factor=1, min_channel_ratio = 1):
        self.decay_factor = 1
        self.min_channel = int(min_channel_ratio*x.shape[1])
        self.channel_num = x.shape[1]
        if self.min_channel <= 0:
            self.min_channel = 1
        self.x = x
        self.y = y
        self.err = 0
        self.errdiff = 0
        self.T = []
    
    def update(self):
        pass

    def getErrGain(self):
        if len(self.T) +self.min_channel >= self.channel_num:
            return float('inf')
        return self.errdiff * decay_factor

'''
collecting training examples for layer pruning
'''
def collecting_training_examples(hook, m=1000):
    x = torch.cat(hook.x, 0)
    del hook
    torch.cuda.empty_cache()
    samplesize = x.shape[0]
    
    m = min(m, samplesize)
    if m != samplesize:
        selected_index = random.sample(range(samplesize), m)
        x = x[selected_index,:].cuda()

    y = torch.sum(x,1,keepdim=True)
    return x, y, m

def getFilterChannelSelections(model, hooks, train_loader, gpu_id,activation_kernel=None, m=1000,decay_factor = 1,min_channel_ratio = 1):
    print("collecting training examples")
    # assert activation_kernel is not None
    conv_nums = get_conv_nums(model)
    '''
    register hooks for each layer
    '''
    for layer_hook in hooks.values():
        for hook in layer_hook.values():
            hook.register()
            
    total_sample = 0
    model.eval()
    for i, train_data in enumerate(train_loader):
        with torch.no_grad():
            train_data['L'] = train_data['L'].cuda()
            model(train_data['L'])
        total_sample += train_data['L'].shape[0]
        print("inference {} samples".format(total_sample))
        for layer_hook in hooks.values():
            for hook in layer_hook.values():
                hook.get_x(gpu_id)
                torch.cuda.empty_cache()

        if total_sample > m:
            break

    '''
    remove hooks
    '''
    for layer_hook in hooks.values():
        for hook in layer_hook.values():
            hook.remove()

    '''
    get training examples
    '''
    filter_channel_selections = {}
    for idx, layer_hook in hooks.items():
        for hook in layer_hook.values():
            x_list, y_list = [], []
            min_sample_num = float('inf')
            tempx, tempy, sample_num = collecting_training_examples(hook, m)
            min_sample_num = min(min_sample_num, sample_num)
            x_list.append(tempx)
            y_list.append(tempy)
        '''
        here concat and sum are both ok.
        scaling or not are optional,too.
        '''
        for i in range(len(x_list)):
            x_list[i] = x_list[i][:min_sample_num,...]
            y_list[i] = y_list[i][:min_sample_num,...]
        x = torch.cat(x_list,0)
        y = torch.cat(y_list,0)
        filter_channel_selections[idx] = FilterChannelSelection(x, y, pow(decay_factor,conv_nums-idx), min_channel_ratio)
    
    return filter_channel_selections


def get_subset(x, y, r, C):
    '''
    x:shape(sample_size, C)
    y:shape(sample_size, 1)
    r is the compression ratio as mentioned in paper;
    C is the channel num of the filter
    '''
    print("computing the prune subset.")

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
                mask = torch.ones(1, C)-torch.sum(torch.eye(C)[tempT,:], dim=0, keepdim=True)
                mask = mask.cuda() # mask shape (1, C-len(tempT))
                diff = y-torch.sum(mask*x, dim=1,keepdim=True)
                total_error = torch.sum(diff**2)
                if total_error < min_value:
                    min_value = total_error
                    tempRes = tempT
        res = tempRes
    return res

def get_layers(model):
    layers = []
    for i in model.modules():
        if isinstance(i, nn.Conv2d):
            layers.append(i)
    return layers

def get_conv_nums(model):
    num = 0
    for i in model.modules():
        if isinstance(i, nn.Conv2d):
            num += 1
    return num

def get_filter_nums(model):
    filter_nums = 0
    all_layers = get_layers(model)
    for layer in all_layers:
        filter_nums += layer.weight.shape[0]
    return filter_nums

def add_hooks(model):
    succeeding_strategy = unet_succeeding_strategy(4)
    preceding_strategy = get_preceding_from_succeding(succeeding_strategy)

    all_layers = get_layers(model)
    hooks = {}
    for idx, layer in enumerate(all_layers):
        succeeding_layer_idx = succeeding_strategy[layer_idx]
        if succeeding_layer_idx == []: # this means the layer is output layer, don't need prune.
            continue
        C = layer.out_channels
        assert C == layer.weight.shape[0] # assure the layer is pruned for the first time.
        layer_hook = {}
        for sl in succeeding_layer_idx:
            precedding_layers = preceding_strategy[sl]

            activation_kernel = list(range(C))

            if len(precedding_layers) > 1:
                kernel_before = 0
                for pl in precedding_layers:
                    if layer_idx > pl:
                        kernel_before += all_layers[pl].weight.shape[0]
                    else:
                        break
                activation_kernel = list(range(kernel_before, kernel_before+C))
            layer_hook[sl] = hookYandX(model, all_layers[sl], activation_kernel)

        hooks[idx] = layer_hook
    return hooks

def thinet_prune_layer(model, train_loader, r, gpu_id, min_channel_ratio, decay_factor, m=1000):
    hooks = add_hooks(model)
    fcs_dict = getFilterChannelSelections(model, hooks, train_loader, gpu_id, m,decay_factor,min_channel_ratio):
    filter_nums = get_filter_nums(model)
    for iter in range(int(filter_nums*(1-r))):
        '''TODO:get the subset'''
        for idx, fcs in fcs_dict.items():
            fcs.getErrGain()
        pass
    
    '''
    after getting the subset, 2 choices: #1,prune it;#2,use the structure to train from scratch.
    '''


    # WARNING:below is the original code, remember to delete it

    prune_subset = get_subset(x, y, r, C)
    assert len(set(prune_subset)) == len(prune_subset) # assure no duplicate element


    '''
    prune the layer
    '''
    print("pruning layer")
    saved_subset = list(set(range(C))-set(prune_subset))
    w, mlis = get_w_by_LSE(x[:,saved_subset], y)
    saved_subset = [saved_subset[i] for i in mlis]
    w = w.unsqueeze(0).unsqueeze(-1) #shape(1,len(saved_subset),1,1)

    layer.weight.data = layer.weight.data[saved_subset, ...]
    assert layer.weight.data.shape[0] == w.shape[1] # filter num should be the same as the element num of w
    layer.weight.grad = None

    if layer.bias is not None:
        layer.bias.data = layer.bias.data[saved_subset]
        layer.bias.grad = None
    
    '''
    prune the succeeding layer
    '''
    print("pruning succeeding layer's kernel")
    for sl in succeeding_layer_idx:
        precedding_layers = preceding_strategy[sl]
        kernel_before = 0
        # this snippet can handle concat more than 2 inputs, not only 2
        if len(precedding_layers) > 1:
            for pl in precedding_layers:
                if layer_idx > pl:
                    kernel_before += all_layers[pl].weight.shape[0]
                else:
                    break
        offset_prune_subset = [kernel_before+i for i in prune_subset]
        all_kernels = all_layers[sl].weight.shape[1]
        kernel_saved_subset = list(set(range(all_kernels))-set(offset_prune_subset))
        saved_weight = all_layers[sl].weight.data[:,kernel_saved_subset,...]

        # scaling it by w
        scaling_kernel = list(range(kernel_before, kernel_before+len(saved_subset)))
        saved_weight[:,scaling_kernel,...] = saved_weight[:,scaling_kernel,...]*w

        all_layers[sl].weight.data = saved_weight
        all_layers[sl].weight.grad = None

    return model
