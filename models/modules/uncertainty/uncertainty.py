import torch
import torch.nn.functional as F


def kl_divergence(p, q):
    return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=1)


def expert_uncertainty(pred_seg, pred_sdf_aux, pred_bnd, T=1.0):
    """
    输入:
        pred_*: logits (B,C,H,W)
    输出:
        U_seg, U_sdf, U_bnd: (B,H,W)
    """
    p_seg = F.softmax(pred_seg / T, dim=1)
    p_sdf = F.softmax(pred_sdf_aux / T, dim=1)
    p_bnd = F.softmax(pred_bnd / T, dim=1)

    p_seg_detach = p_seg.detach()
    p_sdf_detach = p_sdf.detach()
    p_bnd_detach = p_bnd.detach()

    U_seg = 0.5 * (
        kl_divergence(p_seg, p_sdf_detach) +
        kl_divergence(p_seg, p_bnd_detach)
    )

    U_sdf = 0.5 * (
        kl_divergence(p_sdf, p_seg_detach) +
        kl_divergence(p_sdf, p_bnd_detach)
    )

    U_bnd = 0.5 * (
        kl_divergence(p_bnd, p_seg_detach) +
        kl_divergence(p_bnd, p_sdf_detach)
    )

    return U_seg, U_sdf, U_bnd


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
