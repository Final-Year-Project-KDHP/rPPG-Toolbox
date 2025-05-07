# -------------------------------------------------------------------------
# PhysMamba – multi‑task rPPG + SpO₂ with three MoE injections
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fft
from timm.models.layers import trunc_normal_, DropPath        # noqa: F401  (kept in case you use them elsewhere)
from mamba_ssm import Mamba

# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------
def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Exponentiation with better numerical stability for extremely large values."""
    return torch.exp(torch.tanh(x / 400) * 400)


def rounding_sigmoid_approximation(
    x: torch.Tensor, k: float, n_max: int = 100
) -> torch.Tensor:
    """
    Continuous relaxation of the integer `round()` via a bounded sigmoid sum
    (see Huang & Yang 2020 for details).
    """
    x = x.to(torch.float64)
    n_values = torch.arange(-n_max, n_max + 1, dtype=torch.float64, device=x.device)
    x_expanded = x.unsqueeze(-1)
    n_expanded = n_values.view(1, 1, -1)
    term1 = n_expanded / (1 + safe_exp(-k * (x_expanded - n_expanded + 0.5)))
    term2 = n_expanded / (1 + safe_exp(-k * (x_expanded - n_expanded - 0.5)))
    return (term1 - term2).sum(dim=-1, keepdim=True).squeeze(-1)


# -------------------------------------------------------------------------
# Building blocks
# -------------------------------------------------------------------------
class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        self.hidden_dim = dim * mlp_ratio
        # time→hidden
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim,   self.hidden_dim, 1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        # hidden→time
        self.fc2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, dim,   1, bias=False),
            nn.BatchNorm1d(dim),
        )
        # learnable real/imag mixing
        self.scale = 0.02
        self.r = nn.Parameter(self.scale * torch.randn(self.hidden_dim, self.hidden_dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.hidden_dim, self.hidden_dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.hidden_dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.hidden_dim))

    def forward(self, x):
        # x: [B, N, C]  (N=time, C=dim)
        B, N, C = x.shape
        # → [B, C, N] → hidden → [B, hidden_dim, N]
        h = self.fc1(x.transpose(1,2))
        # → [B, N, hidden_dim]
        h = h.transpose(1,2)
        # FFT over time axis
        H = torch.fft.fft(h, dim=1, norm='ortho')
        # mix real/imag
        Hr = F.relu( H.real @ self.r.t() - H.imag @ self.i.t() + self.rb )
        Hi = F.relu( H.imag @ self.r.t() + H.real @ self.i.t() + self.ib )
        H_ = torch.view_as_complex(torch.stack([Hr, Hi], dim=-1))
        # back to time
        h2 = torch.fft.ifft(H_, dim=1, norm='ortho').real
        # → [B, hidden_dim, N] → [B, C, N]
        y = self.fc2(h2.transpose(1,2)).transpose(1,2)
        return y  # [B, N, C]


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels: int, reduction: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attn


class LateralConnection(nn.Module):
    """Fuse Fast → Slow pathway (SlowFast‑style)."""

    def __init__(self, fast_channels: int = 32, slow_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                fast_channels, slow_channels, (3, 1, 1),
                stride=(2, 1, 1), padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        return slow + self.conv(fast)


class CDC_T(nn.Module):
    """Central‑Difference Convolution along the temporal axis (Fan et al., CVPR 2020)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        theta: float = 0.2,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal

        kT = self.conv.weight.size(2)
        if kT <= 1:
            return out_normal

        k0 = self.conv.weight[:, :, 0]
        k2 = self.conv.weight[:, :, 2]
        k_diff = (k0 + k2).sum(dim=(2, 3), keepdim=False)
        k_diff = k_diff[:, :, None, None, None]

        out_diff = F.conv3d(
            x,
            k_diff,
            bias=None,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return out_normal - self.theta * out_diff


class MambaLayer(nn.Module):
    """Mamba‑SSM block flattened across space–time tokens."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            dim, d_state=d_state, d_conv=d_conv, expand=expand, bimamba=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x_flat = x.reshape(b, c, t * h * w).transpose(1, 2)  # (B,THW,C)
        y = self.norm1(x_flat)
        y = self.mamba(y)
        y = self.norm2(x_flat + y)
        return y.transpose(1, 2).view(b, c, t, h, w)


# -------------------------------------------------------------------------
# Cross‑Scale Fuse (three taps, learnable weights via softmax)
# -------------------------------------------------------------------------
class CrossScaleFuse(nn.Module):
    """
    *Learned* or *fixed* per‑location fusion of three resolution taps.
    If `learnable=False`, `manual_coeffs=[w2,w3]` must be given; w1 is
    inferred s.t. w1+w2+w3 = 1.
    """

    def __init__(
        self,
        channels: int,
        learnable: bool = True,
        manual_coeffs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.weight_gen = nn.Conv3d(channels * 3, 3, 1, bias=True)
        else:
            if manual_coeffs is None or len(manual_coeffs) != 2:
                raise ValueError("manual_coeffs must be [w2,w3] when learnable=False")
            w2, w3 = manual_coeffs
            w1 = 1.0 - (w2 + w3)
            coeff = torch.tensor([w1, w2, w3], dtype=torch.float32)
            # register so it moves with .to(device) but is NOT trainable
            self.register_buffer("coeff", coeff.view(1, 3, 1, 1, 1))

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            cat = torch.cat([f1, f2, f3], dim=1)
            logits = self.weight_gen(cat)
            w1, w2, w3 = torch.softmax(logits, dim=1).chunk(3, dim=1)
        else:  # broadcast stored weights
            w1, w2, w3 = self.coeff.split(1, dim=1)
        return w1 * f1 + w2 * f2 + w3 * f3


# -------------------------------------------------------------------------
# Router (lightweight gating between *task* vs *shared* experts)
# -------------------------------------------------------------------------
class Router(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(8, in_channels // 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,2,1,1,1)
        return self.net(x)


# -------------------------------------------------------------------------
# Convenience conv‑block factory
# -------------------------------------------------------------------------
def conv_block(
    c_in: int,
    c_out: int,
    k,
    stride,
    pad,
    bn: bool = True,
    act: str = "relu",
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Conv3d(c_in, c_out, k, stride, pad)]
    if bn:
        layers.append(nn.BatchNorm3d(c_out))
    if act == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif act == "elu":
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


# -------------------------------------------------------------------------
# MoE fusion helper
# -------------------------------------------------------------------------
def fuse_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    expert_task: nn.Module,
    expert_shared: nn.Module,
) -> torch.Tensor:
    probs = torch.softmax(router_logits, dim=1)  # (B,2,1,1,1)
    p_task, p_sh = probs[:, 0:1], probs[:, 1:2]
    return p_task * expert_task(x) + p_sh * expert_shared(x)


# -------------------------------------------------------------------------
# Main network
# -------------------------------------------------------------------------
class PhysMambaMultiTask(nn.Module):
    """
    Video → (rPPG, SpO₂) with three MoE injections and 3‑way cross‑scale fusion.
    """

    def __init__(
        self,
        theta: float = 0.5,
        drop_rate1: float = 0.25,
        drop_rate2: float = 0.5,
        frames: int = 128,
        learnable_balance: bool = False,
        init_lambda: float = 0.5,
        cross_fuse_cfg=None,                # expect a dict‑like node
        k_round: float = 10.0
    ):
        super().__init__()

        if cross_fuse_cfg is None:    # minimal safety‑check
            raise ValueError("cross_fuse_cfg must be passed from the YAML!")

        # keep for later use in forward()
        self.k_round = k_round

        # create fusers
        self.cross_fuse_rppg = CrossScaleFuse(
            48,
            learnable=cross_fuse_cfg.ENABLED,
            manual_coeffs=cross_fuse_cfg.MANUAL_COEFFS,
        )
        self.cross_fuse_spo2 = CrossScaleFuse(
            48,
            learnable=cross_fuse_cfg.ENABLED,
            manual_coeffs=cross_fuse_cfg.MANUAL_COEFFS,
        )

        self.frames = frames

        # ─── Stem ──────────────────────────────────────────────────────
        self.ConvBlock1 = conv_block(3, 16, (1, 5, 5), 1, (0, 2, 2))
        self.ConvBlock2 = conv_block(16, 32, 3, 1, 1)
        self.ConvBlock3 = conv_block(32, 64, 3, 1, 1)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        # Dual‑path split (SlowFast‑like)
        self.ConvBlock4 = conv_block(64, 64, (4, 1, 1), (4, 1, 1), 0)  # slow
        self.ConvBlock5 = conv_block(64, 32, (2, 1, 1), (2, 1, 1), 0)  # fast
        self.ConvBlock6 = conv_block(32, 32, (3, 1, 1), 1, (1, 0, 0), act="elu")

        # Temporal blocks
        self.Block1_slow, self.Block2_slow, self.Block3_slow = [
            self._make_block(64, theta) for _ in range(3)
        ]
        self.Block1_fast, self.Block2_fast, self.Block3_fast = [
            self._make_block(32, theta) for _ in range(3)
        ]

        # Lateral fusions
        self.fuse_1 = LateralConnection(32, 64)
        self.fuse_2 = LateralConnection(32, 64)

        # Upsample & condense
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(96, 48, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(48),
            nn.ELU(inplace=True),
        )
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # Dropouts
        self.drops = nn.ModuleList(
            [nn.Dropout(p) for p in [drop_rate1, drop_rate1, drop_rate2, drop_rate2, drop_rate2, drop_rate2]]
        )

        # ─── MoE injections ───────────────────────────────────────────
        # Scale‑1 (after fuse_1): 64 → 48
        self.router1_rppg = Router(64)
        self.router1_spo2 = Router(64)
        self.expert1_rppg = nn.Conv3d(64, 48, 1)
        self.expert1_spo2 = nn.Conv3d(64, 48, 1)
        self.expert1_shared = nn.Conv3d(64, 48, 1)

        # Scale‑2 (after fuse_2): 64 → 48
        self.router2_rppg = Router(64)
        self.router2_spo2 = Router(64)
        self.expert2_rppg = nn.Conv3d(64, 48, 1)
        self.expert2_spo2 = nn.Conv3d(64, 48, 1)
        self.expert2_shared = nn.Conv3d(64, 48, 1)

        # Pooled (48‑ch map)
        self.routerP_rppg = Router(48)
        self.routerP_spo2 = Router(48)
        self.expertP_rppg = nn.Conv3d(48, 48, 1)
        self.expertP_spo2 = nn.Conv3d(48, 48, 1)
        self.expertP_shared = nn.Conv3d(48, 48, 1)

        # 3‑way cross‑scale fusers
        self.cross_fuse_rppg = CrossScaleFuse(48)
        self.cross_fuse_spo2 = CrossScaleFuse(48)

        # Shared 1×1 mixing
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(48, 48, 1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.hr_head_norm = nn.BatchNorm3d(48)
        self.spo2_head_norm = nn.BatchNorm3d(48)
        self.ConvLast_hr = nn.Conv3d(48, 1, 1)
        self.ConvLast_spo2 = nn.Conv3d(48, 1, 1)

        # ---------- Frequency block for HR branch ----------
        # we'll apply this to the pooled [B,48,frames,1,1] features
        self.freq_ffn_hr = Frequencydomain_FFN(dim=48, mlp_ratio=2)

        # Optional learnable loss‑balance λ
        raw_init = math.log(init_lambda / (1.0 - init_lambda))
        self._lambda_raw = nn.Parameter(
            torch.tensor(raw_init, dtype=torch.float32),
            requires_grad=learnable_balance,
        )

    # -----------------------------------------------------------------
    @property
    def lambda_task(self) -> torch.Tensor:
        """λ ∈ (0,1) for task‑level loss weighting."""
        return torch.sigmoid(self._lambda_raw)

    # -----------------------------------------------------------------
    def _make_block(self, ch: int, theta: float) -> nn.Sequential:
        return nn.Sequential(
            CDC_T(ch, ch, theta=theta),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            MambaLayer(ch),
            ChannelAttention3D(ch, 2),
        )

    # -----------------------------------------------------------------
    def _forward_stem_dualpath(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.MaxpoolSpa(self.ConvBlock1(x))
        x = self.ConvBlock2(x)
        x = self.MaxpoolSpa(self.ConvBlock3(x))
        s_x, f_x = self.ConvBlock4(x), self.ConvBlock5(x)
        return s_x, f_x

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        rppg : (B,T)     predicted waveform
        spo2 : (B,T)     predicted SpO₂ (rounded 0‑100 %)
        λ    : ()        learnable task balance scalar
        """

        b = x.size(0)

        # === Stem ===
        s_x, f_x = self._forward_stem_dualpath(x)
        # print("stem  ➞", s_x.shape, f_x.shape)

        # === Block set 1 ===
        s_x1 = self.drops[0](self.MaxpoolSpa(self.Block1_slow(s_x)))
        f_x1 = self.drops[1](self.MaxpoolSpa(self.Block1_fast(f_x)))
        s_x1 = self.fuse_1(s_x1, f_x1)
        # print("fuse_1 ➞", s_x1.shape)

        # --- MoE @ scale‑1 ---
        fused1_r = fuse_moe(
            s_x1, self.router1_rppg(s_x1), self.expert1_rppg, self.expert1_shared
        )
        fused1_s = fuse_moe(
            s_x1, self.router1_spo2(s_x1), self.expert1_spo2, self.expert1_shared
        )
        fused1_r = F.interpolate(
            fused1_r,
            size=(self.frames, 1, 1),
            mode="trilinear",
            align_corners=False,
        )
        fused1_s = F.interpolate(
            fused1_s,
            size=(self.frames, 1, 1),
            mode="trilinear",
            align_corners=False,
        )

        # === Block set 2 ===
        s_x2 = self.drops[2](self.MaxpoolSpa(self.Block2_slow(s_x1)))
        f_x2 = self.drops[3](self.MaxpoolSpa(self.Block2_fast(f_x1)))
        s_x2 = self.fuse_2(s_x2, f_x2)
        # print("fuse_2 ➞", s_x2.shape)

        # --- MoE @ scale‑2 ---
        fused2_r = fuse_moe(
            s_x2, self.router2_rppg(s_x2), self.expert2_rppg, self.expert2_shared
        )
        fused2_s = fuse_moe(
            s_x2, self.router2_spo2(s_x2), self.expert2_spo2, self.expert2_shared
        )
        fused2_r = F.interpolate(
            fused2_r,
            size=(self.frames, 1, 1),
            mode="trilinear",
            align_corners=False,
        )
        fused2_s = F.interpolate(
            fused2_s,
            size=(self.frames, 1, 1),
            mode="trilinear",
            align_corners=False,
        )

        # === Block set 3 & merge ===
        s_x3 = self.drops[4](self.upsample1(self.Block3_slow(s_x2)))
        f_x3 = self.drops[5](self.ConvBlock6(self.Block3_fast(f_x2)))
        x_fusion = torch.cat([f_x3, s_x3], dim=1)  # (B,96,T/2, H/2, W/2)
        x_final = self.upsample2(x_fusion)
        x_final = self.poolspa(x_final)             # (B,48,frames,1,1)
        # print("pooled ➞", x_final.shape)

        # --- MoE @ pooled ---
        fusedP_r = fuse_moe(
            x_final, self.routerP_rppg(x_final), self.expertP_rppg, self.expertP_shared
        )
        fusedP_s = fuse_moe(
            x_final, self.routerP_spo2(x_final), self.expertP_spo2, self.expertP_shared
        )

        # === Three‑way cross‑scale fusion ===
        feat_r = self.cross_fuse_rppg(fused1_r, fused2_r, fusedP_r)
        feat_s = self.cross_fuse_spo2(fused1_s, fused2_s, fusedP_s)

        # Shared mixing + heads
        feat_r = self.shared_mlp(feat_r)
        feat_s = self.shared_mlp(feat_s)

        # rppg = self.ConvLast_hr(self.hr_head_norm(feat_r)).view(b, self.frames)

                # apply freq-domain FFN along the time axis on feat_r
        # 1) squeeze spatial dims → [B,48,frames]
        x = feat_r.squeeze(-1).squeeze(-1)             # [B,48,T]
        # 2) permute to [B,T,48]
        x = x.permute(0,2,1)
        # 3) spectral FFN → [B,T,48]
        x = self.freq_ffn_hr(x)
        # 4) back to [B,48,T,1,1]
        x = x.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
        # 5) finish with norm & conv
        rppg = self.ConvLast_hr(self.hr_head_norm(x)).view(b, self.frames)


        spo2 = self.ConvLast_spo2(self.spo2_head_norm(feat_s)).view(b, self.frames)
        # spo2 = rounding_sigmoid_approximation(100.0 * torch.sigmoid(spo2), k=10.0)
        spo2 = rounding_sigmoid_approximation(100.0 * torch.sigmoid(spo2), k=self.k_round)

        return rppg, spo2, self.lambda_task
