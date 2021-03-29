from csgd.csgd_pipeline import csgd_iterative
from constants import VGG_ORIGIN_DEPS, LRSchedule
from base_config import get_baseconfig_by_epoch
from constants import unet_succeeding_strategy, unet_pacesetter_dict, unet_origin_deps_flattened
from network.UNet import UNet
from util import loadYaml, parseArgs
from network.UNet import UNet

base_lrs = LRSchedule(base_lr=5e-2, max_epochs=600, lr_epoch_boundaries=[200, 400], lr_decay_factor=0.1, linear_final_lr=None)
csgd_lrs = LRSchedule(base_lr=3e-2, max_epochs=600, lr_epoch_boundaries=[200, 400], lr_decay_factor=0.1, linear_final_lr=None)
# csgd_lrs = LRSchedule(base_lr=1e-2, max_epochs=600, lr_epoch_boundaries=[200, 400], lr_decay_factor=0.1, linear_final_lr=None)

args = parseArgs()
unetcfg, saveName = loadYaml(args.config)

def csgd_unet():
    try_arg = 'slim_5-8'
    network_type = 'unet'
    # dataset_name = 'cifar10'
    dataset_name = 'GoPro'
    base_log_dir = 'csgd_exps/{}_{}_base'.format(network_type, try_arg)
    csgd_log_dir = 'csgd_exps/{}_{}_csgd'.format(network_type, try_arg)
    weight_decay_strength = 1e-4
    batch_size = 64

    unet = UNet(3, 3)

    origin_deps = unet_origin_deps_flattened(unet)

    centri_strength = 3e-3
    # 3 phase iter to slim
    deps_schedule = [[d * 13 // 16 for d in origin_deps],
                     [d * 11 // 16 for d in origin_deps],
                     [d * 5 // 8 for d in origin_deps]]

    succeeding_strategy = unet_succeeding_strategy(4)
    pacesetter_dict = unet_pacesetter_dict()

    base_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=base_lrs.max_epochs, base_lr=base_lrs.base_lr, lr_epoch_boundaries=base_lrs.lr_epoch_boundaries,
                                     lr_decay_factor=base_lrs.lr_decay_factor,
                                     warmup_epochs=5, warmup_method='linear', warmup_factor=0,
                                     ckpt_iter_period=20000, tb_iter_period=100, output_dir=base_log_dir,
                                     tb_dir=base_log_dir, save_weights=None, val_epoch_period=2, linear_final_lr=base_lrs.linear_final_lr,
                                     weight_decay_bias=0, deps=origin_deps)

    csgd_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=csgd_lrs.max_epochs, base_lr=csgd_lrs.base_lr, lr_epoch_boundaries=csgd_lrs.lr_epoch_boundaries,
                                     lr_decay_factor=csgd_lrs.lr_decay_factor,
                                     warmup_epochs=5, warmup_method='linear', warmup_factor=0,
                                     ckpt_iter_period=20000, tb_iter_period=100, output_dir=csgd_log_dir,
                                     tb_dir=csgd_log_dir, save_weights=None, val_epoch_period=2, linear_final_lr=csgd_lrs.linear_final_lr,
                                     weight_decay_bias=0, deps=origin_deps)
    csgd_iterative(model=unet, init_weights=None, base_train_config=base_config,
                   csgd_train_config=csgd_config,
                   itr_deps=deps_schedule, centri_strength=centri_strength, pacesetter_dict=pacesetter_dict,
                   succeeding_strategy=succeeding_strategy, extra_cfg=unetcfg)

if __name__ == '__main__':
    csgd_unet()
