# ------------------------------------------------------
# Frequencydomain_FFN ‑‑ stand‑alone version reused from RhythmMamba
# ------------------------------------------------------
import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from timm.models.layers import trunc_normal_, lecun_normal_
from einops import rearrange


class Frequencydomain_FFN(nn.Module):
    """
    FFT → learnable complex mixing → IFFT   (per‑feature spectral MLP)
    Input : (B,N,C)   Output : (B,N,C)
    """

    def __init__(self, dim: int, mlp_ratio: int = 2):
        super().__init__()
        self.scale = 0.02
        self.dim_h = dim * mlp_ratio   # hidden ‑ (F in the paper)

        # learnable *complex* weights  (r,i)  and biases (rb,ib)
        self.r  = nn.Parameter(self.scale * torch.randn(self.dim_h, self.dim_h))
        self.i  = nn.Parameter(self.scale * torch.randn(self.dim_h, self.dim_h))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim_h))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim_h))

        # point‑wise conv projections
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, self.dim_h, 1, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(self.dim_h, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
        )

        self._init_weights()

    # --------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,N,C)     (N = temporal length)
        """
        B, N, C = x.shape

        # time‑wise point conv ─────────────────────────
        x = self.fc1(x.transpose(1, 2))          # (B,C_h,N)
        x = x.transpose(1, 2)                    # (B,N,C_h)

        # FFT over N  ─────────────────────────────────
        X = torch.fft.fft(x, dim=1, norm='ortho')   # complex64

        # complex linear + ReLU on real & imag parts
        xr = F.relu(torch.einsum('bnf,fg->bng',  X.real, self.r) -
                    torch.einsum('bnf,fg->bng',  X.imag, self.i) + self.rb)
        xi = F.relu(torch.einsum('bnf,fg->bng',  X.imag, self.r) +
                    torch.einsum('bnf,fg->bng',  X.real, self.i) + self.ib)

        X_new = torch.view_as_complex(torch.stack([xr, xi], dim=-1))  # (B,N,C_h)
        x = torch.fft.ifft(X_new, dim=1, norm='ortho').real           # back to ℝ

        # project back & return ──────────────────────
        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)               # (B,N,C)
        return x
