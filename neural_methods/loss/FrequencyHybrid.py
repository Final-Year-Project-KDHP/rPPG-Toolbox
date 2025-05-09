# ── neural_methods/loss/FrequencyHybrid.py  (REPLACEMENT) ───────────────────
import torch, math
import torch.nn.functional as F
from typing import Union, Dict, Tuple
from evaluation.POST_PROCESS import calculate_hr
from .torchlosscomputer_fine import PSDProjector


# --------------------------------------------------------------------------- #
#                          Σ  exp(-(x-μ)²/2σ²)  helper                        #
# --------------------------------------------------------------------------- #
def _gaussian_pmf(centre: torch.Tensor,
                  bpm_vec: torch.Tensor,
                  std: float = 3.0,
                  eps: float = 1e-12) -> torch.Tensor:
    """Row‑wise Gaussian (broadcastable)."""
    g = torch.exp(-0.5 * ((bpm_vec - centre.unsqueeze(1)) / std) ** 2)
    g = torch.clamp(g, min=eps)
    return g / g.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
#                       Finer, vectorised frequency loss                      #
# --------------------------------------------------------------------------- #
def frequency_loss_waveform_fine(
        pred_wave: torch.Tensor,          # (B,T)
        gt_wave  : torch.Tensor,          # (B,T)
        projector: PSDProjector,
        *,
        std: float = 3.0,                 # σ of the KL Gaussian (bpm)
        tau: float = 4.0,                 # temperature for soft regression
        scale_ce: float = 60.0,           # **NEW**: scale raw p before CE
        w_ce: float = 8.0,
        w_kl: float = 4.0,
        w_reg: float = 3.0,
        w_harm: float = 0.25,
        eps_bpm: float = 8.0,
        r_max: float = 0.45,
        eps: float = 1e-9
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Same spirit as the original function but **uses raw probabilities p**
    (scaled by `scale_ce`) instead of log‑probs everywhere Cross‑Entropy
    is required.  KL still needs log p, computed on‑the‑fly.
    """
    device = pred_wave.device
    B      = pred_wave.size(0)

    # ---------- PSD projection ------------------------------------------------
    p = projector(pred_wave)                       # (B,F) rows sum to 1
    logits = p * scale_ce                          # treat as logits for CE
                                                  #  (scaling restores dynamic range)

    logp = torch.log(p + eps)                      # still needed for KL

    # ---------- Ground‑truth HR bins -----------------------------------------
    hr_gt = torch.tensor([
        calculate_hr(pred_wave[i].detach().cpu(),
                     gt_wave [i].detach().cpu(),
                     diff_flag=False, fs=projector.Fs)[1]
        for i in range(B)
    ], device=device)                                            # (B,)

    idx_fund = ((hr_gt - 45.0) / projector.step).round().long()  # (B,)
    idx_fund = idx_fund.clamp(0, p.size(1) - 1)

    idx_harm = (2 * idx_fund).clamp(0, p.size(1) - 1)

    # ---------- CE -----------------------------------------------------------
    loss_ce = F.cross_entropy(logits, idx_fund)

    # ---------- KL divergence -----------------------------------------------
    bpm_vec = projector.k * 60.0                                  # (F,)
    target  = _gaussian_pmf(hr_gt.float(), bpm_vec, std=std, eps=eps)
    loss_kl = F.kl_div(logp, target, reduction='batchmean')

    # ---------- Soft regression ---------------------------------------------
    probs   = F.softmax(logits / tau, dim=1)
    exp_hr  = (probs * bpm_vec).sum(dim=1)
    loss_reg = F.l1_loss(exp_hr, hr_gt.float(), reduction='mean')

    # ---------- 2nd‑harmonic suppression ------------------------------------
    bins_per_bpm = 1.0 / projector.step
    eps_bins     = int(round(eps_bpm * bins_per_bpm))

    range_idx = torch.arange(p.size(1), device=device)
    fund_mask = ((range_idx[None] >= (idx_fund - eps_bins).unsqueeze(1)) &
                 (range_idx[None] <= (idx_fund + eps_bins).unsqueeze(1))).float()
    harm_mask = ((range_idx[None] >= (idx_harm - eps_bins).unsqueeze(1)) &
                 (range_idx[None] <= (idx_harm + eps_bins).unsqueeze(1))).float()

    p_fund = (p * fund_mask).sum(dim=1)
    p_harm = (p * harm_mask).sum(dim=1)
    ratio_loss = F.relu(p_harm / (p_fund + eps) - r_max).mean()

    # ---------- Combine ------------------------------------------------------
    total = (w_ce   * loss_ce +
             w_kl   * loss_kl +
             w_reg  * loss_reg +
             w_harm * ratio_loss)

    return total, dict(
        ce   = loss_ce.item(),
        kl   = loss_kl.item(),
        reg  = loss_reg.item(),
        ratio= ratio_loss.item()
    )
