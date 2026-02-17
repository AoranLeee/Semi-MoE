import torch
import torch.nn as nn


class DWConv(nn.Module):
    """
    Basic Depthwise Convolution block.

    Args:
        in_channels (int)
        kernel_size (int)
        use_bn (bool)
        activation (str): 'relu' | 'gelu' | None
    """

    def __init__(
        self,
        in_channels,
        kernel_size=3,
        use_bn=True,
        activation="relu",
    ):
        super().__init__()
        padding = kernel_size // 2

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
