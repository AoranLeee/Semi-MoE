import torch
import torch.nn.functional as F


def symmetric_kl_uncertainty(logits, logits_e):
    """
    logits:    (B, C, H, W)
    logits_e:  (B, C, H, W)
    """
    p = F.softmax(logits, dim=1)
    p_e = F.softmax(logits_e, dim=1)

    p_bar = 0.5 * (p + p_e)

    eps = 1e-8
    term1 = torch.log((p_bar + eps) / (p + eps))
    term2 = torch.log((p_bar + eps) / (p_e + eps))

    U = p_bar * (term1 + term2)
    U = torch.sum(U, dim=1)

    return U


def sdf_uncertainty(logits, logits_e):
    """
    logits:    (B, 1, H, W)
    logits_e:  (B, 1, H, W)
    """
    U = (torch.tanh(logits) - torch.tanh(logits_e)) ** 2
    return U.squeeze(1)
