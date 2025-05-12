# neural_methods/loss/FFTDistance.py
import torch
import torch.nn.functional as F
from typing import Tuple, Union

def fft_band_distance(
        pred_wave: torch.Tensor,      # (B,T)
        gt_wave:   torch.Tensor,      # (B,T)  – only to get HR label if you wish
        fs: float  = 30.0,
        pad_N: int = 512,
        f_low: float = 0.75,          # 45 bpm
        f_high: float = 2.5,          # 150 bpm
        p: int = 1,                   # L1 (1) or L2 (2) norm on magnitude
        power: bool = False,
        eps: float = 1e-8,
        r_max: float = 0.15           # harmonic‑ratio constraint
) -> Tuple[torch.Tensor, dict]:
    """
    Differentiable spectral loss  L = ‖ |F_pred| – |F_gt| ‖_p  (band‑limited)
    + harmonic penalty on P₂f/P_f.
    """
    device = pred_wave.device
    B, T   = pred_wave.shape
    window = torch.hann_window(T, device=device)

    # 1) rFFT (with zero‑pad) -----------------------------
    def _mag(x):
        spec = torch.fft.rfft(x*window, n=pad_N, dim=-1)
        mag  = spec.abs()            # (B, F_pad)
        return mag[:, 1:]            # drop DC bin

    mag_p = _mag(pred_wave)
    mag_g = _mag(gt_wave).detach()   # detach label PSD

    F_bins = mag_p.size(-1)
    freqs  = torch.linspace(0, fs/2, F_bins+1, device=device)[1:]  # skip DC

    band = (freqs >= f_low) & (freqs <= f_high)
    mag_p_band = mag_p[:, band]
    mag_g_band = mag_g[:, band]

    # 2) magnitude distance --------------------------------
    if power:
        mag_p_band = mag_p_band.pow(2)
        mag_g_band = mag_g_band.pow(2)

    # normalise each PSD so Σ=1 (prevents scale explosion)
    mag_p_band = mag_p_band / (mag_p_band.sum(-1, keepdim=True)+eps)
    mag_g_band = mag_g_band / (mag_g_band.sum(-1, keepdim=True)+eps)

    loss_mag = F.l1_loss(mag_p_band, mag_g_band, reduction='mean') if p==1 \
             else F.mse_loss(mag_p_band, mag_g_band, reduction='mean')

    # 3) harmonic penalty  ---------------------------------
    # locate f₀ bin from GT PSD (coarse but cheap)
    k0 = mag_g_band.argmax(-1)                         # (B,)
    k2 = (2 * k0).clamp(max=mag_p_band.size(-1)-1)    # second harmonic
    P1 = mag_p_band[torch.arange(B), k0]
    P2 = mag_p_band[torch.arange(B), k2]
    loss_h = F.relu(P2 / (P1 + eps) - r_max).mean()

    total = loss_mag + loss_h
    return total, dict(mag=loss_mag.item(), harm=loss_h.item())
