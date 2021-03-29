# What each function does in C-SGD

## constants.py

- rc_origin_deps_flattened()

  返回一个list,按顺序包含每个conv的filter数量，即outchannel

- rc_convert_flattened_deps()

  返回一个list，用于生成各个stage的ResBlock

- rc_succeeding_strategy()

  返回一个dict，每个conv的输出对应的conv，即每个conv的直接子layer

- rc_pacesetter_dict()

  返回一个dict，每个follower对应的pacesetter。形成一个祖先树一样的结构。根据论文，即每个树中节点都对应同样的剪枝策略

- 