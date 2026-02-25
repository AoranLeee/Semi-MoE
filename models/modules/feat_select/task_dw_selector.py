import torch
import torch.nn as nn
import numpy as np
from .dwconv import DWConv


# def sigmoid_rampup(current: int, rampup_length: int) -> float:
#     """
#     Sigmoid ramp-up function used for alpha scheduling.
#     """
#     if rampup_length == 0:
#         return 1.0
#     current = np.clip(current, 0.0, rampup_length)
#     phase = 1.0 - current / rampup_length
#     return float(np.exp(-5.0 * phase * phase))


class TaskDWSelector(nn.Module):
    """
    Scale-wise Task-specific DWConv selector.

    Args:
        in_channels (int)
        num_tasks (int)
        return_weight (bool):
            True -> return weight map
            False -> return reweighted feature
        detach_input (bool): whether to stop-grad encoder feature
    """

    def __init__(
        self,
        in_channels,
        num_tasks,
        return_weight=False,
        detach_input=False,
        kernel_size=3,
        use_bn=True,
        activation="relu",
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.return_weight = return_weight
        self.detach_input = detach_input
        self.alpha = 0.0

        self.task_dw = nn.ModuleList([
            DWConv(
                in_channels,
                kernel_size=kernel_size,
                use_bn=use_bn,
                activation=activation,
            )
            for _ in range(num_tasks)
        ])

        self.sigmoid = nn.Sigmoid()
        self.last_weight_maps = []
        self.last_var = None

    def clear_last_weight_maps(self):
        self.last_weight_maps = []

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def get_weight_stats(self):
        """
        Return statistics of last forward weight maps.
        Returns:
            dict with keys:
                mean
        """
        if not self.last_weight_maps:
            return None

        stats = {}
        for idx, weight in enumerate(self.last_weight_maps):
            stats[f"task{idx}"] = {
                "mean": weight.mean().item(),
            }
        return stats

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns:
            list length = num_tasks
        """
        if self.detach_input:
            x = x.detach()

        outputs = []
        self.last_weight_maps = []
        self.last_var = None
        gates = []
        for dw in self.task_dw:
            logits = dw(x)
            gate = self.sigmoid(logits)
            gates.append(gate)

        if len(gates) == 0:
            return outputs

        g = torch.cat(gates, dim=1)
        g_mean = g.mean(dim=1, keepdim=True)
        g_rel = g - g_mean
        g_spatial = g.mean(dim=(2, 3))
        g_spatial_mean = g_spatial.mean(dim=1, keepdim=True)
        self.last_var = ((g_spatial - g_spatial_mean) ** 2).mean()

        self.last_weight_maps = list(gates)

        for t in range(self.num_tasks):
            g_rel_t = g_rel[:, t:t + 1, :, :]
            if self.return_weight:
                outputs.append(g_rel_t)
            else:
                outputs.append(x * (1 + self.alpha * g_rel_t))
        return outputs
