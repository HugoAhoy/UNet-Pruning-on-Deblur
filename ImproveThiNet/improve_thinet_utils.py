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
    def __init__(self, x, y, gpu_id,decay_factor=1, min_channel_ratio = 1):
        assert decay_factor != 0
        self.decay_factor = decay_factor
        self.min_channel = int(min_channel_ratio*x.shape[1])
        self.channel_num = x.shape[1]
        if self.min_channel <= 0:
            self.min_channel = 1
        self.x = x.cuda(gpu_id)
        self.y = y.cuda(gpu_id)
        self.gpu_id = gpu_id
        self.err = 0
        self.errdiff = 0
        self.tempErr = 0
        self.T = []
        self.tempT = []
        self.update()
    
    def update(self):
        if len(self.T) +self.min_channel >= self.channel_num:
            self.errdiff = float('inf')

        # greedy
        self.T = self.tempT
        self.err =self.tempErr
        min_value = float('inf')
        tempRes = None
        for i in range(self.channel_num):
            if i in self.T:
                continue
            else:
                tempT = self.T + [i]
                mask = torch.ones(1, self.channel_num)-torch.sum(torch.eye(self.channel_num)[tempT,:], dim=0, keepdim=True)
                mask = mask.cuda(self.gpu_id) # mask shape (1, C-len(tempT))
                diff = self.y-torch.sum(mask*self.x, dim=1,keepdim=True)
                total_error = torch.sum(diff**2)
                if total_error < min_value:
                    min_value = total_error
                    tempRes = tempT
        self.tempT = tempRes
        self.tempErr = min_value
        self.errdiff = self.tempErr - self.err

    def getErrGain(self):
        return self.errdiff * self.decay_factor
    
    def getRank(self):
        saved_subset = list(set(range(self.channel_num))-set(self.T))
        return torch.matrix_rank(self.x[:,saved_subset])
    
    def getSavedChannelNum(self):
        return self.channel_num - len(self.T)

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

def getFilterChannelSelections(model, hooks, train_loader, gpu_id, m=1000,decay_factor = 1,min_channel_ratio = 1):
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

        if total_sample >= m:
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
        filter_channel_selections[idx] = FilterChannelSelection(x, y, gpu_id,pow(decay_factor,idx), min_channel_ratio)
    
    return filter_channel_selections



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
        succeeding_layer_idx = succeeding_strategy[idx]
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
                    if idx > pl:
                        kernel_before += all_layers[pl].weight.shape[0]
                    else:
                        break
                activation_kernel = list(range(kernel_before, kernel_before+C))
            layer_hook[sl] = hookYandX(all_layers[sl], activation_kernel)

        hooks[idx] = layer_hook
    return hooks

def improve_thinet_pruned_structure(model, train_loader, r, gpu_id, min_channel_ratio, decay_factor, m=1000):
    hooks = add_hooks(model)
    fcs_dict = getFilterChannelSelections(model, hooks, train_loader, gpu_id, m,decay_factor,min_channel_ratio)
    filter_nums = get_filter_nums(model)
    print("total filter num is {}".format(filter_nums))
    
    '''get the subset'''
    for iter in range(int(filter_nums*(1-r))):
        min_val = float('inf')
        selected_idx = -1
        for idx, fcs in fcs_dict.items():
            errGain = fcs.getErrGain()
            if errGain < min_val:
                min_val = errGain
                selected_idx = idx
        print("iter:{}, select {}".format(iter, selected_idx))
        fcs_dict[selected_idx].update()
    
    '''
    after getting the subset, 2 choices: #1,prune it;#2,use the structure to train from scratch.
    but because the previous experiment prove that on deblur task, the conclusion in "rethinking ..." 
    still works. So here I straight forward use the pruned stuctrue to train from scratch, 
    rather than using the weight to fine-tune.
    And based the found in LSE, here use rank to furthe squeeze the model width.
    '''

    '''
    TODO:generate the structure from fcs_dict
    '''
    # raw version
    all_layers = get_layers(model)
    filter_dict = {}
    for i, layer in enumerate(all_layers):
        filter_dict[i] = layer.weight.shape[0]
    
    for idx, fcs in fcs_dict.items():
        filter_dict[idx] = fcs.getSavedChannelNum()
    
    # use rank to further reduce width
    filter_dict_with_rank = {}
    for i, layer in enumerate(all_layers):
        filter_dict_with_rank[i] = layer.weight.shape[0]
    
    for idx, fcs in fcs_dict.items():
        filter_dict_with_rank[idx] = fcs.getRank()
    
    return filter_dict, filter_dict_with_rank