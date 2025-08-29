import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from skimage.morphology import dilation, disk


def sdf_loss(pred, sdf):
    # pred = pred * 2
    criterion = nn.MSELoss()
    return criterion(pred, sdf)

def identify_axis(shape):
    """
    Helper function to enable loss function to be flexibly used for 
    both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
    """
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]
    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

def to_onehot(y_pred, y_true):
    shp_x = y_pred.shape
    shp_y = y_true.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(y_pred.shape, y_true.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y_true 
        else:
            y_true = y_true.long()
            y_onehot = torch.zeros(shp_x, device=y_pred.device)
            y_onehot.scatter_(1, y_true, 1)
    return y_onehot

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, square=False, weight=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    y_onehot = to_onehot(net_output, gt)

    if weight is None:
        weight = 1
    tp = net_output * y_onehot * weight
    fp = net_output * (1 - y_onehot) * weight
    fn = (1 - net_output) * y_onehot * weight
    tn = (1 - net_output) * (1 - y_onehot) * weight

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

def imbalance_diceLoss(y_pred, y_true, smooth=1e-8):
        # first convert y_true to one-hot format
    axis = identify_axis(y_pred.shape)
    y_pred = nn.Softmax(dim=1)(y_pred)
    tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis)
    intersection = 2 * tp + smooth
    union = 2 * tp + fp + fn + smooth
    dice = 1 - (intersection / union)
    return dice.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.sigma2 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.sigma3 = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, loss1, loss2, loss3):
        sigma1 = self.sigma1
        sigma2 = self.sigma2
        sigma3 = self.sigma3

        loss = (
            torch.exp(-sigma1) * loss1 + 0.4 * torch.exp(sigma1) +
            0.5 * torch.exp(-sigma2) * loss2 + 0.4 * torch.exp(sigma2) + 
            torch.exp(-sigma3) * loss3 + 0.4 * torch.exp(sigma3)
        )
        return loss
 