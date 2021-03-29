# from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
# from model_map import get_model_fn
# from torch.nn.modules.loss import CrossEntropyLoss
from utils.engine import Engine
from utils.pyt_utils import ensure_dir
# from utils.misc import torch_accuracy, AvgMeter
# from collections import OrderedDict
import torch
# import time
# from utils.lr_scheduler import get_lr_scheduler
import os
from visdom import Visdom
import numpy as np
# from ding_test import run_eval
from csgd.csgd_prune import csgd_prune_and_save
from sklearn.cluster import KMeans
from dataset.gopro import GoProDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import dataset.util as util


TRAIN_SPEED_START = 0.1
TRAIN_SPEED_END = 0.2

KERNEL_KEYWORD = 'conv.weight'

def add_vecs_to_mat_dicts(param_name_to_merge_matrix):
    kernel_names = set(param_name_to_merge_matrix.keys())
    for name in kernel_names:
        bias_name = name.replace(KERNEL_KEYWORD, 'conv.bias')
        gamma_name = name.replace(KERNEL_KEYWORD, 'bn.weight')
        beta_name = name.replace(KERNEL_KEYWORD, 'bn.bias')
        param_name_to_merge_matrix[bias_name] = param_name_to_merge_matrix[name]
        param_name_to_merge_matrix[gamma_name] = param_name_to_merge_matrix[name]
        param_name_to_merge_matrix[beta_name] = param_name_to_merge_matrix[name]


def generate_merge_matrix_for_kernel(deps, layer_idx_to_clusters, kernel_namedvalue_list):
    result = {}
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        merge_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            if len(clst) == 1:
                merge_trans_mat[clst[0], clst[0]] = 1
                continue
            sc = sorted(clst)       # Ideally, clst should have already been sorted in ascending order
            for ei in sc:
                for ej in sc:
                    merge_trans_mat[ei, ej] = 1 / len(clst)
        result[kernel_namedvalue_list[layer_idx].name] = torch.from_numpy(merge_trans_mat).cuda()
    return result

def generate_decay_matrix_for_kernel_and_vecs(deps, layer_idx_to_clusters, kernel_namedvalue_list, weight_decay, centri_strength):
    result = {}
    #   for the kernel
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                decay_trans_mat[ee, ee] = weight_decay + centri_strength
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength / len(clst)
        kernel_mat = torch.from_numpy(decay_trans_mat).cuda()
        result[kernel_namedvalue_list[layer_idx].name] = kernel_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.bias')] = kernel_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'conv.bias')] = kernel_mat

    #   for the vec params (bias, beta and gamma), we use 0.1 * centripetal strength
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                # Note: using smaller centripetal strength on the scaling factor of BN improve the performance in some of the cases
                decay_trans_mat[ee, ee] = weight_decay + centri_strength * 0.1
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength * 0.1 / len(clst)
        vec_mat = torch.from_numpy(decay_trans_mat).cuda()

        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.weight')] = vec_mat

    return result

def cluster_by_kmeans(kernel_value, num_cluster):
    assert kernel_value.ndim == 4
    x = np.reshape(kernel_value, (kernel_value.shape[0], -1))
    if num_cluster == x.shape[0]:
        result = [[i] for i in range(num_cluster)]
        return result
    else:
        print('cluster {} filters into {} clusters'.format(x.shape[0], num_cluster))
    km = KMeans(n_clusters=num_cluster)
    km.fit(x)
    result = []
    for j in range(num_cluster):
        result.append([])
    for i, c in enumerate(km.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result

def _is_follower(layer_idx, pacesetter_dict):
    followers_and_pacesetters = set(pacesetter_dict.keys())
    return (layer_idx in followers_and_pacesetters) and (pacesetter_dict[layer_idx] != layer_idx)

def get_layer_idx_to_clusters(kernel_namedvalue_list, target_deps, pacesetter_dict):
    result = {}
    for layer_idx, named_kv in enumerate(kernel_namedvalue_list):
        num_filters = named_kv.value.shape[0]
        if pacesetter_dict is not None and _is_follower(layer_idx, pacesetter_dict):
            continue
        if num_filters > target_deps[layer_idx]:
            result[layer_idx] = cluster_by_kmeans(kernel_value=named_kv.value, num_cluster=target_deps[layer_idx])
        elif num_filters < target_deps[layer_idx]:
            raise ValueError('wrong target dep')
    return result

def train_one_step(net, data, groudtruth, optimizer, criterion, param_name_to_merge_matrix, param_name_to_decay_matrix):
    pred = net(data)
    loss = criterion(pred, groudtruth)
    loss.backward()

    for name, param in net.named_parameters():
        if name in param_name_to_merge_matrix:
            p_dim = param.dim()
            p_size = param.size()
            if p_dim == 4:
                param_mat = param.reshape(p_size[0], -1)
                g_mat = param.grad.reshape(p_size[0], -1)
            elif p_dim == 1:
                param_mat = param.reshape(p_size[0], 1)
                g_mat = param.grad.reshape(p_size[0], 1)
            else:
                assert p_dim == 2
                param_mat = param
                g_mat = param.grad

            csgd_gradient = param_name_to_merge_matrix[name].matmul(g_mat) + param_name_to_decay_matrix[name].matmul(param_mat)
            param.grad.copy_(csgd_gradient.reshape(p_size))

    optimizer.step()
    optimizer.zero_grad()
    return loss

def sgd_optimizer(cfg, model, use_nesterov):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.weight_decay_bias
            print('set weight_decay_bias={} for {}'.format(weight_decay, key))
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        params += [{"params": [value], "lr": apply_lr, "weight_decay": 0}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer


def get_optimizer(cfg, model, use_nesterov=False):
    return sgd_optimizer(cfg, model, use_nesterov=use_nesterov)

def get_criterion(cfg):
    return torch.nn.L1Loss()

def csgd_train_and_prune(cfg:BaseConfigByEpoch,
                        target_deps, centri_strength, pacesetter_dict, succeeding_strategy, pruned_weights,extra_cfg,
                         net=None, train_dataloader=None, val_dataloader=None, show_variables=False, beginning_msg=None,
               init_weights=None, no_l2_keywords=None, use_nesterov=False, tensorflow_style_init=False,iter=None):

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    clusters_save_path = os.path.join(cfg.output_dir, 'clusters.npy')
    print("cluster save path:{}".format(clusters_save_path))
    config = extra_cfg

    with Engine() as engine:

        is_main_process = (engine.world_rank == 0) #TODO correct?

        logger = engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        saveName = "%s-%s.yaml" % (config['note'], config['dataset'])
        modelName = config['modelName']
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_available']
        device_ids = range(config['gpu_num'])
        trainSet = GoProDataset(sharp_root=config['train_sharp'], blur_root=config['train_blur'],
                                resize_size=config['resize_size'], patch_size=config['crop_size'],
                                phase='train')
        testSet = GoProDataset(sharp_root=config['test_sharp'], blur_root=config['test_blur'],
                            resize_size=config['resize_size'], patch_size=config['crop_size'],
                            phase='test')

        train_loader = DataLoader(trainSet,
                                batch_size=config['batchsize'],
                                shuffle=True, num_workers=4,
                                drop_last=True, pin_memory=True)
        test_loader = DataLoader(testSet, batch_size=1,
                                shuffle=False, num_workers=1,
                                drop_last=False, pin_memory=True)

        print('NOTE: Data prepared')
        print('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(config['batchsize'], torch.cuda.device_count(), torch.cuda.memory_allocated()))
    
        model = net
        
        optimizer = get_optimizer(cfg, model, use_nesterov=use_nesterov)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
        criterion = get_criterion(cfg).cuda()

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer, cfg=cfg)

        model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

        # load weight of last prune iteration or the not pruned model
        if init_weights:
            engine.load_pth(init_weights)

        # for unet the last outconv will not be pruned
        kernel_namedvalue_list = engine.get_all_conv_kernel_namedvalue_as_list(remove = 'out')

        # cluster filters
        if os.path.exists(clusters_save_path):
            layer_idx_to_clusters = np.load(clusters_save_path,allow_pickle=True).item()
            print("cluster exist, load from {}".format(clusters_save_path))
        else:
            layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                              target_deps=target_deps, pacesetter_dict=pacesetter_dict)
            if pacesetter_dict is not None:
                for follower_idx, pacesetter_idx in pacesetter_dict.items():
                    if pacesetter_idx in layer_idx_to_clusters:
                        layer_idx_to_clusters[follower_idx] = layer_idx_to_clusters[pacesetter_idx]
            
            # print(layer_idx_to_clusters)

            np.save(clusters_save_path, layer_idx_to_clusters)

        csgd_save_file = os.path.join(cfg.output_dir, 'finish.pth')

        # if this prune iter has a trained model, then load it
        if os.path.exists(csgd_save_file):
            engine.load_pth(csgd_save_file)
        else:
            param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=cfg.deps,
                                                                          layer_idx_to_clusters=layer_idx_to_clusters,
                                                                          kernel_namedvalue_list=kernel_namedvalue_list)
            param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(deps=cfg.deps,
                                                                          layer_idx_to_clusters=layer_idx_to_clusters,
                                                                          kernel_namedvalue_list=kernel_namedvalue_list,
                                                                          weight_decay=cfg.weight_decay,
                                                                          centri_strength=centri_strength)
            # if pacesetter_dict is not None:
            #     for follower_idx, pacesetter_idx in pacesetter_dict.items():
            #         follower_kernel_name = kernel_namedvalue_list[follower_idx].name
            #         pacesetter_kernel_name = kernel_namedvalue_list[follower_idx].name
            #         if pacesetter_kernel_name in param_name_to_merge_matrix:
            #             param_name_to_merge_matrix[follower_kernel_name] = param_name_to_merge_matrix[
            #                 pacesetter_kernel_name]
            #             param_name_to_decay_matrix[follower_kernel_name] = param_name_to_decay_matrix[
            #                 pacesetter_kernel_name]

            # add 2 para of bn and conv.bias to mat dicts to enable the c-sgd update rule
            add_vecs_to_mat_dicts(param_name_to_merge_matrix)

            if show_variables:
                engine.show_variables()

            if beginning_msg:
                engine.log(beginning_msg)

            logger.info("\n\nStart training with pytorch version {}".format(torch.__version__))

            iteration = engine.state.iteration
            startEpoch = config['start_epoch']
            max_epochs =config['max_epochs']

            engine.save_pth(os.path.join(cfg.output_dir, 'init.pth'))

            viz = Visdom(env=saveName)
            bestPSNR = config['bestPSNR']

            itr = '' if iter is None else str(iter)
            for epoch in range(startEpoch, max_epochs):
                # eval
                if epoch % config['save_epoch'] == 0:
                    with torch.no_grad():
                        model.eval()
                        avg_PSNR = 0
                        idx = 0
                        for test_data in test_loader:
                            idx += 1
                            test_data['L'] = test_data['L'].cuda()
                            sharp = model(test_data['L'])
                            sharp = sharp.detach().float().cpu()
                            sharp = util.tensor2uint(sharp)
                            test_data['H'] = util.tensor2uint(test_data['H'])
                            current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                            avg_PSNR += current_psnr
                            if idx % 100 == 0:
                                print("epoch {}: tested {}".format(epoch, idx))
                        avg_PSNR = avg_PSNR / idx
                        print("total PSNR : {:<4.2f}".format(
                            avg_PSNR))
                        viz.line(
                            X=[epoch],
                            Y=[avg_PSNR],
                            win='testPSNR-'+itr,
                            opts=dict(title='psnr', legend=['valid_psnr']),
                            update='append')
                        if avg_PSNR > bestPSNR:
                            bestPSNR = avg_PSNR
                            save_path = os.path.join(cfg.output_dir, 'finish.pth')
                            engine.save_pth(save_path)

                # train
                avg_loss = 0.0
                idx = 0
                model.train()
                for i, train_data in enumerate(train_loader):
                    idx += 1
                    train_data['L'] = train_data['L'].cuda()
                    train_data['H'] = train_data['H'].cuda()
                    optimizer.zero_grad()
                    loss = train_one_step(model, train_data['L'], train_data['H'], criterion,\
                                     optimizer,param_name_to_merge_matrix,\
                                     param_name_to_decay_matrix)

                    avg_loss += loss.item()
                    if idx % 100 == 0:
                        print("epoch {}: trained {}".format(epoch, idx))

                scheduler.step()
                avg_loss = avg_loss / idx
                print("epoch {}: total loss : {:<4.2f}, lr : {}".format(
                    epoch, avg_loss, scheduler.get_lr()[0]))
                viz.line(
                    X=[epoch],
                    Y=[avg_loss],
                    win='trainMSELoss-'+itr,
                    opts=dict(title='mse', legend=['train_mse']),
                    update='append')
            # engine.save_pth(os.path.join(cfg.output_dir, 'finish.pth'))

        csgd_prune_and_save(engine=engine, layer_idx_to_clusters=layer_idx_to_clusters,
                            save_file=pruned_weights, succeeding_strategy=succeeding_strategy, new_deps=target_deps)