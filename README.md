# Intro

In this repo, I implemented all algorithms involved in my graduation design project about model pruning at image deblurring.

Each algorithm is well organized in directories at root path.

now， there are following pruning algorithms:

- `csgd` : An implementation for Centripetal SGD pruning
- `ThiNet` : An implementation for ThiNet pruning. I improved the ThiNet. The original version ThiNet can't prune the shortcut structure( including skip connection and residul connection). My modification version can handle shortcut structure. Thus, modificated ThiNet can achieve high prune ratio even on networks with shortcut whose depths are not that deep(like 20 to 30 layers).
- `ScaledNet` : This is for comparison,  [*Rethinking the Value of Network Pruning*](https://arxiv.org/abs/1810.05270) think the structure of pruning is more important than the pruned weight. So here the Scaled Net can train the pruned structure from scratch, for comparing experiments.
