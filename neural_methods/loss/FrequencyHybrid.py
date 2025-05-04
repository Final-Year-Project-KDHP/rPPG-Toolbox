# neural_methods/loss/FrequencyHybrid.py

import torch
import torch.nn.functional as F
from .TorchLossComputer import TorchLossComputer
from evaluation.POST_PROCESS import calculate_hr
from typing import Union

def _gaussian_pmf(center_bpm: float,
                  bpm_range: torch.Tensor,
                  std: float = 3.0,
                  eps: float = 1e-12):
    g = torch.exp(-0.5 * ((bpm_range - center_bpm) / std) ** 2)
    g = torch.clamp(g, min=eps)
    return g / g.sum()

def frequency_loss_waveform(
    pred_wave: torch.Tensor,   # [B, T]
    gt_wave  : torch.Tensor,   # [B, T]
    Fs       : Union[int, float],
    diff_flag: bool,
    std      : float = 3.0, #3
    tau      : float = 1.5, #2
    w_ce     : float = 50,#0.3
    w_kl     : float = 25,#0.3
    w_reg    : float = 50 #0.4,8
):
    """
    Hybrid frequency loss using the built-in normalized PSD from complex_absolute.
    """
    device = pred_wave.device
    B, T   = pred_wave.shape

    # 1) Compute ground-truth HR per example (detached)
    hr_gt_list = []
    for i in range(B):
        _, hr_g = calculate_hr(
            pred_wave[i].detach().cpu(),
            gt_wave [i].detach().cpu(),
            diff_flag=diff_flag,
            fs=Fs
        )
        hr_gt_list.append(hr_g)
    hr_gt = torch.tensor(hr_gt_list, device=device, dtype=torch.long)  # [B]

    # 2) Build candidate BPM bins
    bpm_range = torch.arange(45, 150, device=device, dtype=pred_wave.dtype)  # [105]

    # 3) Compute normalized PSD “pmf” directly
    #    complex_absolute(...) already returns P_i / sum_j P_j
    ca_list = []
    for i in range(B):
        ca_i = TorchLossComputer.complex_absolute(
            pred_wave[i].view(1, -1),  # [1,T]
            Fs,
            bpm_range
        )  # returns shape [1,105], sums to 1
        ca_list.append(ca_i)
    p = torch.cat(ca_list, dim=0)    # [B,105], each row sums to 1

    # 4) Negative log-likelihood on the true bin
    #    F.cross_entropy expects logits; we pass log(p + eps) so that
    #      log_softmax(log p) = log p - log sum(exp(log p)) = log p  (since sum p =1).
    eps = 1e-12
    logits = torch.log(p + eps)      # [B,105]
    loss_ce = F.cross_entropy(logits, (hr_gt - 45).clamp(0,104))

    # 5) KL divergence against a Gaussian soft target
    target = torch.stack([
        _gaussian_pmf(g, bpm_range, std=std, eps=eps)  for g in hr_gt.float()
    ], dim=0)                      # [B,105]
    # use F.kl_div with log-prob inputs
    loss_kl = F.kl_div(logits, target, reduction='batchmean')

    # 6) Optional expected-HR regression
    #    soften via temperature, if desired
    probs = F.softmax(logits / tau, dim=1)   # [B,105]
    exp_hr = (probs * bpm_range).sum(dim=1)  # [B]
    loss_reg = F.l1_loss(exp_hr, hr_gt.float(), reduction='mean')

    # 7) Combine
    total = w_ce * loss_ce + w_kl * loss_kl + w_reg * loss_reg
    return total, {
        'ce':   loss_ce.item(),
        'kl':   loss_kl.item(),
        'reg':  loss_reg.item(),
        'hr_gt': hr_gt_list
    }
