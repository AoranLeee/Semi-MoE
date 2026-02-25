from .task_dw_selector import TaskDWSelector
import torch
import torch.nn as nn


class MultiScaleTaskSelector(nn.Module):
    """
    Scale-wise task-aware feature selection.

    Args:
        in_channels_list (list[int]): channels for [f1,...,f5]
        num_tasks (int): default 3 (seg/sdf/bnd)
        mode (str): 'task_dw' or 'hybrid'
        hybrid_scales (iter[int] | None): scale indices to leave unselected in hybrid mode
        return_weight (bool): pass-through to TaskDWSelector
        detach_input (bool): pass-through to TaskDWSelector
    """

    def __init__(
        self,
        in_channels_list,
        num_tasks=3,
        mode="task_dw",
        hybrid_scales=None,
        return_weight=False,
        detach_input=False,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.mode = mode
        self.hybrid_scales = set(hybrid_scales or [])
        self.lambda_var = 1e-3
        self.last_var_mean = None
        self.last_loss_var = None
        self.last_var_per_scale = {}

        self.selectors = nn.ModuleList()
        self.selector_map = []
        self.scale_selectors = None
        for idx, in_ch in enumerate(in_channels_list):
            use_task_dw = self.mode == "task_dw" or idx not in self.hybrid_scales
            if use_task_dw:
                # Plugin point: replace TaskDWSelector with expert-based selector.
                selector = TaskDWSelector(
                    in_channels=in_ch,
                    num_tasks=num_tasks,
                    return_weight=return_weight,
                    detach_input=detach_input,
                )
                self.selectors.append(selector)
                self.selector_map.append(selector)
            else:
                # Hybrid scale: placeholder for future expert selector.
                self.selector_map.append(None)

        self.scale_selectors = self.selector_map

    def forward(self, features):
        # features: list [f1, f2, f3, f4, f5]
        if len(features) != len(self.selector_map):
            raise ValueError("features length must match in_channels_list length")

        task_features = [[] for _ in range(self.num_tasks)]
        var_list = []
        self.last_var_per_scale = {}
        for scale_idx, (feat, selector) in enumerate(zip(features, self.selector_map)):
            if selector is None:
                # Hybrid scale: pass-through for now
                for t in range(self.num_tasks):
                    task_features[t].append(feat)
            else:
                outputs = selector(feat)
                if selector.last_var is not None:
                    var_list.append(selector.last_var)
                    self.last_var_per_scale[scale_idx] = selector.last_var
                for t, out in enumerate(outputs):
                    task_features[t].append(out)

        if len(var_list) > 0:
            var_mean = torch.stack(var_list).mean()
            self.last_var_mean = var_mean
            self.last_loss_var = -self.lambda_var * var_mean
        else:
            self.last_var_mean = None
            self.last_loss_var = None
        return task_features #长度为3，每个元素是一个list，包含5个scale的特征

    def get_all_weight_stats(self):
        stats = {}

        for scale_idx, selector in enumerate(self.scale_selectors):
            if selector is None:
                continue
            scale_stats = selector.get_weight_stats()
            if not scale_stats:
                continue
            for task_name, task_stat in scale_stats.items():
                key = f"scale{scale_idx}_{task_name}"
                stats[key] = task_stat

        return stats
