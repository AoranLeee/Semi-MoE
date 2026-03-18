from .task_dw_selector import TaskDWSelector, LowRankExpertSelector
import torch.nn as nn


class MultiScaleTaskSelector(nn.Module):
    """
    Scale-wise task-aware feature selection.

    Args:
        in_channels_list (list[int]): channels for [f1,...,f5]
        num_tasks (int): default 3 (seg/sdf/bnd)
        mode (str): 'task_dw' | 'hybrid' | 'expert'
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

        self.selectors = nn.ModuleList()
        self.selector_map = []
        self.scale_selectors = None
        for idx, in_ch in enumerate(in_channels_list):
            use_task_dw = self.mode == "task_dw" or idx not in self.hybrid_scales
            if use_task_dw:
                if mode == "expert":
                    selector = LowRankExpertSelector(
                        in_channels=in_ch,
                        num_tasks=num_tasks,
                        K=4,
                        weak_prior=True,
                        detach_input=detach_input,
                    )
                else:
                    selector = TaskDWSelector(
                        in_channels=in_ch,
                        num_tasks=num_tasks,
                        return_weight=return_weight,
                        detach_input=detach_input,
                    )
                self.selectors.append(selector)
                self.selector_map.append(selector)
            else:
                # Hybrid scale: pass-through for now.
                self.selector_map.append(None)

        self.scale_selectors = self.selector_map

    def forward(self, features, detach_selector=False):
        # features: list [f1, f2, f3, f4, f5]
        if len(features) != len(self.selector_map):
            raise ValueError("features length must match in_channels_list length")

        task_features = [[] for _ in range(self.num_tasks)]
        for feat, selector in zip(features, self.selector_map):
            if selector is None:
                for t in range(self.num_tasks):
                    task_features[t].append(feat)
                continue

            if self.mode == "expert":
                outputs = selector(feat, detach_selector=detach_selector)
            else:
                outputs = selector(feat, alpha=selector.alpha, detach_selector=detach_selector)

            for t, out in enumerate(outputs):
                task_features[t].append(out)

        return task_features

    def get_all_weight_stats(self):
        stats = {}
        for scale_idx, selector in enumerate(self.selector_map):
            if selector is None:
                continue
            scale_stats = selector.get_weight_stats()
            if scale_stats is None:
                continue
            for name, value in scale_stats.items():
                key = f"scale{scale_idx}_{name}"
                stats[key] = {"value": value}

        return stats
