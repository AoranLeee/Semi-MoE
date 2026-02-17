import torch
import torch.nn as nn
from .dwconv import DWConv


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

    def clear_last_weight_maps(self):
        self.last_weight_maps = []

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns:
            list length = num_tasks
        """
        if self.detach_input:
            x = x.detach()

        outputs = []
        weight_maps = []

        for dw in self.task_dw:
            weight = self.sigmoid(dw(x))
            weight_maps.append(weight)
            if self.return_weight:
                outputs.append(weight)
            else:
                outputs.append(weight * x)

        self.last_weight_maps = weight_maps
        return outputs
