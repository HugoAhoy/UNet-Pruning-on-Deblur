from baseTrain import trainUNet
import os
# from ding_test import general_test
from csgd.csgd_train import csgd_train_and_prune

def csgd_prune_pipeline(model, init_weights, base_train_config, csgd_train_config,
                        target_deps, centri_strength, pacesetter_dict, succeeding_strategy, extra_cfg,iter):
    #   If there is no given base weights file, train from scratch.
    if init_weights is None:
        if not os.path.exists(base_train_config.output_dir):
            os.mkdir(base_train_config.output_dir)
        csgd_init_weights = os.path.join(base_train_config.output_dir, 'finish.pth')
        if not os.path.exists(csgd_init_weights):
            print("train from scratch")
            trainUNet(extra_cfg, csgd_init_weights)
    else:
        csgd_init_weights = init_weights

    #   C-SGD train then prune
    pruned_weights = os.path.join(csgd_train_config.output_dir, 'pruned.pth')
    csgd_train_and_prune(net=model,cfg=csgd_train_config,
                        target_deps=target_deps, centri_strength=centri_strength,
                         pacesetter_dict=pacesetter_dict, succeeding_strategy=succeeding_strategy,
                         pruned_weights=pruned_weights,
                         init_weights=csgd_init_weights, use_nesterov=True,extra_cfg=extra_cfg,iter=iter)  # TODO init?

#     #   Test it.
#     general_test(csgd_train_config.network_type, weights=pruned_weights)


def csgd_iterative(model, init_weights, base_train_config, csgd_train_config,
                        itr_deps, centri_strength, pacesetter_dict, succeeding_strategy, extra_cfg):

    for itr, deps in enumerate(itr_deps):
        # !!!!!!!! for test, only iter 2 time. comment the statement below after testing
        if itr >= 1:
            break

        print("Iteration:{}".format(itr))
        if itr == 0:
            begin_weights = init_weights
        else:
            begin_weights = os.path.join(csgd_train_config.output_dir, 'itr{}'.format(itr-1), 'pruned.pth')
        
        # if have pruned, go to next prune iteration
        itr_output_dir = os.path.join(csgd_train_config.output_dir, 'itr{}'.format(itr))
        if os.path.exists(os.path.join(itr_output_dir, 'pruned.pth')):
            continue

        itr_csgd_config = csgd_train_config._replace(tb_dir=itr_output_dir)._replace(output_dir=itr_output_dir)

        # change the conv deps in accordance with iter
        if itr != 0:
            itr_csgd_config = itr_csgd_config._replace(deps=itr_deps[itr-1])
        csgd_prune_pipeline(model=model,init_weights=begin_weights, base_train_config=base_train_config,
                            csgd_train_config=itr_csgd_config, target_deps=deps,
                            centri_strength=centri_strength, pacesetter_dict=pacesetter_dict,
                            succeeding_strategy=succeeding_strategy, extra_cfg=extra_cfg,iter=itr)

