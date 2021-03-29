from collections import OrderedDict
import numpy as np
from utils.misc import save_hdf5
import torch

def save_pth(pthdict, path):
    save_dict = {}
    new_state_dict = OrderedDict()
    num_params = 0
    for k, v in pthdict['model'].items():
        key = k
        if k.split('.')[0] == 'module':
            key = k[7:]
        v = torch.from_numpy(v).type(torch.float32)
        new_state_dict[key] = v
        num_params += v.numel()
    save_dict['model'] = new_state_dict

    if 'deps' in pthdict:
        save_dict['deps'] = pthdict['deps']
    torch.save(save_dict, path)
    print('save {} values to {} after pruning'.format(len(save_dict['model']), path))

def delete_or_keep(array, idxes, axis=None):
    if len(idxes) > 0:
        return np.delete(array, idxes, axis=axis)
    else:
        return array

def parse_succeeding_strategy(layer_idx_to_clusters, succeeding_strategy):
    if succeeding_strategy is None:
        succeeding_map = None
    elif succeeding_strategy == 'simple':
        succeeding_map = {idx : (idx+1) for idx in layer_idx_to_clusters.keys()}
    else:
        succeeding_map = succeeding_strategy
    return succeeding_map

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

def csgd_prune_and_save(engine, layer_idx_to_clusters, save_file, succeeding_strategy, new_deps):
    result = OrderedDict()

    succeeding_map = parse_succeeding_strategy(succeeding_strategy=succeeding_strategy, layer_idx_to_clusters=layer_idx_to_clusters)
    preceding_map = get_preceding_from_succeding(succeeding_map)
    print(preceding_map)

    kernel_namedvalues = engine.get_all_kernel_namedvalue_as_list()
    
    for i in kernel_namedvalues:
        print(i.name)

    for layer_idx, namedvalue in enumerate(kernel_namedvalues):
        if layer_idx not in layer_idx_to_clusters:
            continue

        k_name = namedvalue.name
        k_value = namedvalue.value
        if k_name in result:                    # If this kernel has been pruned because it is subsequent to another layer
            k_value = result[k_name]

        clusters = layer_idx_to_clusters[layer_idx]

        #   Prune the kernel
        idx_to_delete = []
        for clst in clusters:
            # only save the 1st filter in each cluster
            if len(clst) > 1:
                idx_to_delete += clst[1:]
            else:
                continue

        kernel_value_pruned = delete_or_keep(k_value, idx_to_delete, axis=0)
        print('cur kernel name: {}, from {} to {}'.format(k_name, k_value.shape, kernel_value_pruned.shape))
        result[k_name] = kernel_value_pruned
        assert new_deps[layer_idx] == kernel_value_pruned.shape[0]

        #   Prune the related vector params
        def handle_vecs(key_name):
            vec_name = k_name.replace('conv.weight', key_name)
            vec_value = engine.get_param_value_by_name(vec_name)
            if vec_value is not None:
                vec_value_pruned = delete_or_keep(vec_value, idx_to_delete)
                result[vec_name] = vec_value_pruned

        handle_vecs('conv.bias')
        handle_vecs('bn.weight')
        handle_vecs('bn.bias')
        handle_vecs('bn.running_mean')
        handle_vecs('bn.running_var')

        #   Handle the succeeding kernels
        # if no succeeding layer, continue
        if layer_idx not in succeeding_map:
            continue

        follows = succeeding_map[layer_idx]
        print('{} follows {}'.format(follows, layer_idx))
        if type(follows) is not list:
            follows = [follows]
        
        for follow_idx in follows:
            follow_kernel_value = kernel_namedvalues[follow_idx].value
            follow_kernel_name = kernel_namedvalues[follow_idx].name
            if follow_kernel_name in result:
                follow_kernel_value = result[follow_kernel_name]
            print('following kernel name: ', follow_kernel_name, 'origin shape: ', follow_kernel_value.shape)
        
            if follow_kernel_value.ndim == 2:   # The following is a FC layer
                fc_idx_to_delete = []
                num_filters = k_value.shape[0]
                fc_neurons_per_conv_kernel = follow_kernel_value.shape[1] // num_filters
                print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
             
                for clst in clusters:
                    if len(clst) == 1:
                        continue
                    for i in clst[1:]:
                        fc_idx_to_delete.append(np.arange(i * fc_neurons_per_conv_kernel,
                                                          (i+1) * fc_neurons_per_conv_kernel))
                    to_concat = []
                    for i in clst:
                        corresponding_neurons_idx = np.arange(i * fc_neurons_per_conv_kernel,
                                                          (i+1) * fc_neurons_per_conv_kernel)
                        to_concat.append(np.expand_dims(follow_kernel_value[:, corresponding_neurons_idx], axis=0))
                    summed = np.sum(np.concatenate(to_concat, axis=0), axis=0)
                    reserved_idx = np.arange(clst[0] * fc_neurons_per_conv_kernel, (clst[0]+1) * fc_neurons_per_conv_kernel)
                    follow_kernel_value[:, reserved_idx] = summed
                if len(fc_idx_to_delete) > 0:
                    follow_kernel_value = delete_or_keep(follow_kernel_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
                result[follow_kernel_name] = follow_kernel_value
                print('shape of pruned following kernel: ', follow_kernel_value.shape)
            
            elif follow_kernel_value.ndim == 4:     # The following is a conv layer
                channelup = 0
                for i in preceding_map[follow_idx]:
                    if i < layer_idx:
                        channelup += len(layer_idx_to_clusters[layer_idx])
                    else:
                        break
                print("channel above of layer {} after layer {}:{}".format(follow_idx, layer_idx, channelup))
                for clst in clusters:
                    clst = [lidx+channelup for lidx in clst]
                    selected_k_follow = follow_kernel_value[:, clst, :, :]
                    summed_k_follow = np.sum(selected_k_follow, axis=1)
                    follow_kernel_value[:, clst[0], :, :] = summed_k_follow
                idx_to_delete = [idx+channelup for idx in idx_to_delete]
                follow_kernel_value = delete_or_keep(follow_kernel_value, idx_to_delete, axis=1)
                result[follow_kernel_name] = follow_kernel_value
                print('shape of pruned following kernel: ', follow_kernel_value.shape)
            else:
                raise ValueError('wrong ndim of kernel')
        print("-------------------------------------")

    key_variables = engine.state_values()
    for name, value in key_variables.items():
        if name not in result:
            result[name] = value
    save_dict ={'model':result}
    save_dict['deps'] = new_deps

    save_pth(save_dict, save_file)
