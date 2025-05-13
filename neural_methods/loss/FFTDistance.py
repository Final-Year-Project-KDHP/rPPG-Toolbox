# ---------------------------------------------------------------------
# Fully differentiable FFT‑based loss for rPPG
#   ‑ magnitude distance in HR band
#   ‑ harmonic‑ratio penalty
#   ‑ softmax‑expectation regression toward GT HR
# ---------------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Tuple, Union

def fft_band_distance(
        pred_wave: torch.Tensor,       # (B,T)
        gt_wave:   torch.Tensor,       # (B,T) – ONLY used for magnitude term
        fs:  float = 30.0,
        pad_N: int = 512,              # zero‑pad length (power‑of‑2)
        f_low: float = 0.75, f_high: float = 2.5,  # band in Hz
        p_norm: int = 1,               # 1=L1, 2=L2
        use_power: bool = False,
        eps: float = 1e-8,
        # ---- harmonic control
        r_max: float = 0.15,
        # ---- soft‑expectation regression
        gt_hr_bpm: Union[torch.Tensor, None] = None,   # (B,)
        tau: float = 2.0,
        w_reg: float = 0.5
) -> Tuple[torch.Tensor, dict]:
    """
    Return: total_loss, aux_dict
    """
    device = pred_wave.device
    B, T   = pred_wave.shape
    window = torch.hann_window(T, device=device)

    # ----------  rFFT  ------------------------------------------------
    def _mag(x):
        spec = torch.fft.rfft(x * window, n=pad_N, dim=-1)
        return spec.abs()[:, 1:]           # drop DC bin

    mag_p = _mag(pred_wave)               # (B,F_pad)
    mag_g = _mag(gt_wave).detach()        # label spectrum, no grad

    F_bins = mag_p.size(-1)
    freqs  = torch.linspace(0, fs/2, F_bins+1, device=device)[1:]  # Hz/bin

    band_mask = (freqs >= f_low) & (freqs <= f_high)
    mag_p_b = mag_p[:, band_mask]
    mag_g_b = mag_g[:, band_mask]

    if use_power:
        mag_p_b, mag_g_b = mag_p_b.pow(2), mag_g_b.pow(2)

    # normalise row‑wise
    mag_p_b = mag_p_b / (mag_p_b.sum(-1, keepdim=True) + eps)
    mag_g_b = mag_g_b / (mag_g_b.sum(-1, keepdim=True) + eps)

    # ---------- magnitude distance -----------------------------------
    if p_norm == 1:
        loss_mag = F.l1_loss(mag_p_b, mag_g_b, reduction='mean')
    else:
        loss_mag = F.mse_loss(mag_p_b, mag_g_b, reduction='mean')

    # ---------- harmonic‑ratio penalty -------------------------------
    idx0 = mag_g_b.argmax(-1)                              # coarse f₀ bin
    idx2 = (2*idx0).clamp(max=mag_p_b.size(-1)-1)
    P1 = mag_p_b[torch.arange(B, device=device), idx0]
    P2 = mag_p_b[torch.arange(B, device=device), idx2]
    loss_h = F.relu(P2 / (P1 + eps) - r_max).mean()

    # ---------- softmax‑expectation regression -----------------------
    if gt_hr_bpm is not None:
        gt_f = gt_hr_bpm.to(device) / 60.0                 # Hz
        log_p = torch.log(mag_p_b + eps)
        q     = torch.softmax(log_p / tau, dim=-1)         # sharpened pdf
        exp_f = (q * freqs[band_mask]).sum(-1)             # (B,)
        loss_reg = F.l1_loss(exp_f, gt_f, reduction='mean')
    else:
        loss_reg = mag_p_b.new_tensor(0.0)

    total = loss_mag + w_reg * loss_reg
    aux   = dict(mag=loss_mag.item(),
                 harm=loss_h.item(),
                 reg=loss_reg.item())
    return total, aux
