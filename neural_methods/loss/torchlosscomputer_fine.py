# ── torchlosscomputer_fine.py ──────────────────────────────────────
import torch, math, numpy as np
from torch import nn
from torch.autograd import Function

class PSDProjector(nn.Module):
    """
    Project a (B,T) time‑domain tensor onto an arbitrary list of
    frequencies using vectorised sin/cos dot‑products.
    Keeps its basis in a registered buffer for speed.
    """
    def __init__(self, Fs: float, bpm_low=45, bpm_high=150, step=0.1):
        super().__init__()
        # build k vector (1×F)
        bpm = torch.arange(bpm_low, bpm_high, step)          # (F,)
        hz  = bpm / 60.0
        self.register_buffer("k", hz)                        # non‑trainable
        self.Fs = Fs
        self.basis_T = None                                  # lazy built

    def _build_basis(self, T: int, device):
        n = torch.arange(T, device=device).float()           # (T,)
        omega = 2 * math.pi * self.k[:, None] / self.Fs      # (F,1) broadcast
        sin = torch.sin(omega * n)                           # (F,T)
        cos = torch.cos(omega * n)
        self.basis_T = torch.stack([sin, cos], dim=0)        # (2,F,T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,T)  →  power (B,F) already normalised to sum=1 per row
        """
        B, T = x.shape
        if (self.basis_T is None) or (self.basis_T.shape[-1] != T):
            self._build_basis(T, x.device)

        sin, cos = self.basis_T            # each (F,T)

        # window – optional Hann
        window = torch.hann_window(T, device=x.device)
        xw = x * window                    # (B,T)

        # dot products
        re = torch.einsum('bt,ft->bf', xw, cos)   # (B,F)
        im = torch.einsum('bt,ft->bf', xw, sin)   # (B,F)
        power = re.pow(2) + im.pow(2)             # (B,F)
        power = power / power.sum(dim=1, keepdim=True)
        return power                              # pmf (B,F)
