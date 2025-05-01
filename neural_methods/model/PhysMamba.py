import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba

################################################################################
# Utility helpers (same as before)
################################################################################

def safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.tanh(x / 400) * 400)


def rounding_sigmoid_approximation(x: torch.Tensor, k: float, n_max: int = 100) -> torch.Tensor:
    x = x.to(torch.float64)
    n_values = torch.arange(-n_max, n_max + 1, dtype=torch.float64, device=x.device)
    x_expanded = x.unsqueeze(-1)
    n_expanded = n_values.view(1, 1, -1)
    term1 = n_expanded / (1 + safe_exp(-k * (x_expanded - n_expanded + 0.5)))
    term2 = n_expanded / (1 + safe_exp(-k * (x_expanded - n_expanded - 0.5)))
    return (term1 - term2).sum(dim=-1, keepdim=True).squeeze(-1)

################################################################################
# Shared building blocks (unchanged apart from new Cross‑Scale fuse)
################################################################################

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attn


class LateralConnection(nn.Module):
    def __init__(self, fast_channels=32, slow_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )

    def forward(self, slow, fast):
        return slow + self.conv(fast)


class CDC_T(nn.Module):
    """Central‑Difference Convolution along the temporal axis."""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, theta=0.2):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.theta = theta

    def forward(self, x):
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
            x, k_diff, bias=None,
            stride=self.conv.stride, padding=0,
            dilation=self.conv.dilation, groups=self.conv.groups
        )
        return out_normal - self.theta * out_diff


class MambaLayer(nn.Module):
    """Thin wrapper around mamba‑ssm with token‑wise processing."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mamba = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand, bimamba=True)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x_flat = x.reshape(b, c, t * h * w).transpose(1, 2)
        y = self.norm1(x_flat)
        y = self.mamba(y)
        y = self.norm2(x_flat + y)
        return y.transpose(1, 2).view(b, c, t, h, w)

################################################################################
# New: Cross‑Scale Fuse (learned soft mask across depths)
################################################################################
class CrossScaleFuse(nn.Module):
    """Learned per‑location soft weighting between two depth taps (scale‑2 vs pooled).

    Given two feature maps with identical shape (B, C, T, H, W), the module
    outputs a weighted sum where the weights are predicted by a 1×1×1 conv over
    the concatenated features. Softmax along the *scale* dimension enforces
    w_scale2 + w_pooled = 1 at every spatio‑temporal position.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight_gen = nn.Conv3d(channels * 2, 2, kernel_size=1, bias=True)

    def forward(self, feat_scale2: torch.Tensor, feat_pooled: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([feat_scale2, feat_pooled], dim=1)  # B, 2C, T, H, W
        logits = self.weight_gen(cat)                       # B, 2, T, H, W
        weights = torch.softmax(logits, dim=1)             # ensure sum=1
        w2, wP = weights[:, 0:1], weights[:, 1:2]
        return w2 * feat_scale2 + wP * feat_pooled

################################################################################
# New: light Router module (borrowed from MLoRE SpatialAtt)
################################################################################

class Router(nn.Module):
    """Outputs 2 logits – task vs shared – for its caller."""
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(8, in_channels // 4), 1),
            nn.ReLU(),
            nn.Conv3d(max(8, in_channels // 4), 2, 1)
        )
    def forward(self, x):
        return self.net(x)          # (B,2,1,1,1)

################################################################################
# Small helper to build standard conv blocks
################################################################################

def conv_block(c_in, c_out, k, stride, pad, bn=True, act='relu'):
    layers = [nn.Conv3d(c_in, c_out, k, stride, pad)]
    if bn:
        layers.append(nn.BatchNorm3d(c_out))
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

################################################################################
# MoE fusion helper (unchanged)
################################################################################

def fuse_moe(x, router_logits, expert_task, expert_shared):
    probs = torch.softmax(router_logits, dim=1)  # B,2,1,1,1
    p_task, p_sh = probs[:, 0:1], probs[:, 1:2]
    return p_task * expert_task(x) + p_sh * expert_shared(x)

################################################################################
# Main Network
################################################################################

class PhysMambaMultiTask(nn.Module):
    """Backbone + two MoE injections + cross‑scale weighting + two shallow heads."""

    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128,learnable_balance: bool = False,init_lambda: float = 0.5):
        super().__init__()
        self.frames = frames
        # Stem convs
        self.ConvBlock1 = conv_block(3, 16, (1, 5, 5), 1, (0, 2, 2))
        self.ConvBlock2 = conv_block(16, 32, 3, 1, 1)
        self.ConvBlock3 = conv_block(32, 64, 3, 1, 1)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        # Dual‑path split
        self.ConvBlock4 = conv_block(64, 64, (4, 1, 1), (4, 1, 1), 0)  # slow
        self.ConvBlock5 = conv_block(64, 32, (2, 1, 1), (2, 1, 1), 0)  # fast
        self.ConvBlock6 = conv_block(32, 32, (3, 1, 1), 1, (1, 0, 0), act='elu')

        # Temporal blocks
        self.Block1_slow, self.Block2_slow, self.Block3_slow = [self._make_block(64, theta) for _ in range(3)]
        self.Block1_fast, self.Block2_fast, self.Block3_fast = [self._make_block(32, theta) for _ in range(3)]

        # Lateral fuses
        self.fuse_1 = LateralConnection(32, 64)
        self.fuse_2 = LateralConnection(32, 64)

        # Upsample & channel condense
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64), nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(96, 48, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(48), nn.ELU(),
        )
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # Dropouts
        self.drops = nn.ModuleList([nn.Dropout(p) for p in [drop_rate1, drop_rate1, drop_rate2, drop_rate2, drop_rate2, drop_rate2]])

        # =====================  MoE additions  ===================== #
        # Scale‑2 routers & experts (64‑ch)
        self.router2_rppg = Router(64)
        self.router2_spo2 = Router(64)
        self.expert2_rppg = nn.Conv3d(64, 48, 1)
        self.expert2_spo2 = nn.Conv3d(64, 48, 1)
        self.expert2_shared = nn.Conv3d(64, 48, 1)

        # Pooled routers & experts (48‑ch)
        self.routerP_rppg = Router(48)
        self.routerP_spo2 = Router(48)
        self.expertP_rppg = nn.Conv3d(48, 48, 1)
        self.expertP_spo2 = nn.Conv3d(48, 48, 1)
        self.expertP_shared = nn.Conv3d(48, 48, 1)

        # Cross‑scale fusers (learn soft mask)
        self.cross_fuse_rppg = CrossScaleFuse(48)
        self.cross_fuse_spo2 = CrossScaleFuse(48)

        # Shared channel‑mixing 1×1 conv
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(48, 48, 1, bias=False), nn.BatchNorm3d(48), nn.ReLU(inplace=True)
        )

        # Heads
        self.hr_head_norm, self.spo2_head_norm = nn.BatchNorm3d(48), nn.BatchNorm3d(48)
        self.ConvLast_hr = nn.Conv3d(48, 1, 1)
        self.ConvLast_spo2 = nn.Conv3d(48, 1, 1)

        raw_init = math.log(init_lambda / (1.0 - init_lambda))
        self._lambda_raw = nn.Parameter(
            torch.tensor(raw_init, dtype=torch.float32),
            requires_grad=learnable_balance
        )

    @property
    def lambda_task(self) -> torch.Tensor:
        # map unconstrained raw → (0,1)
        return torch.sigmoid(self._lambda_raw)
        

    # ---------------------------------------------------------------------
    def _make_block(self, ch, theta):
        return nn.Sequential(
            CDC_T(ch, ch, theta=theta), nn.BatchNorm3d(ch), nn.ReLU(),
            MambaLayer(ch), ChannelAttention3D(ch, 2)
        )

    # ---------------------------------------------------------------------
    def _forward_stem_dualpath(self, x):
        x = self.MaxpoolSpa(self.ConvBlock1(x))
        x = self.ConvBlock2(x)
        x = self.MaxpoolSpa(self.ConvBlock3(x))
        s_x, f_x = self.ConvBlock4(x), self.ConvBlock5(x)
        return s_x, f_x

    # ---------------------------------------------------------------------
    def forward(self, x):
        """Return rPPG (B,T) and SpO₂ (B,T)."""
        b = x.size(0)

        # === Stem & dual path === #
        s_x, f_x = self._forward_stem_dualpath(x)

        # === Block set 1 === #
        s_x1 = self.drops[0](self.MaxpoolSpa(self.Block1_slow(s_x)))
        f_x1 = self.drops[1](self.MaxpoolSpa(self.Block1_fast(f_x)))
        s_x1 = self.fuse_1(s_x1, f_x1)

        # === Block set 2 === #
        s_x2 = self.drops[2](self.MaxpoolSpa(self.Block2_slow(s_x1)))
        f_x2 = self.drops[3](self.MaxpoolSpa(self.Block2_fast(f_x1)))
        s_x2 = self.fuse_2(s_x2, f_x2)

        # -----------------  MoE injection @ Scale‑2 ----------------- #
        log2_r = self.router2_rppg(s_x2)
        log2_s = self.router2_spo2(s_x2)
        fused2_r = fuse_moe(s_x2, log2_r, self.expert2_rppg, self.expert2_shared)
        fused2_s = fuse_moe(s_x2, log2_s, self.expert2_spo2, self.expert2_shared)
        fused2_r = F.interpolate(fused2_r, size=(self.frames, 1, 1), mode='trilinear', align_corners=False)
        fused2_s = F.interpolate(fused2_s, size=(self.frames, 1, 1), mode='trilinear', align_corners=False)

        # === Block set 3 + fusion === #
        s_x3 = self.drops[4](self.upsample1(self.Block3_slow(s_x2)))
        f_x3 = self.drops[5](self.ConvBlock6(self.Block3_fast(f_x2)))
        x_fusion = torch.cat([f_x3, s_x3], dim=1)
        x_final = self.upsample2(x_fusion)
        x_final = self.poolspa(x_final)              # (B,48,128,1,1)

        # -----------------  MoE injection @ Pooled ----------------- #
        logP_r = self.routerP_rppg(x_final)
        logP_s = self.routerP_spo2(x_final)
        fusedP_r = fuse_moe(x_final, logP_r, self.expertP_rppg, self.expertP_shared)
        fusedP_s = fuse_moe(x_final, logP_s, self.expertP_spo2, self.expertP_shared)

        # -----------------  Cross‑scale weighting ----------------- #
        feat_r = self.cross_fuse_rppg(fused2_r, fusedP_r)
        feat_s = self.cross_fuse_spo2(fused2_s, fusedP_s)

        # Shared channel mix
        feat_r = self.shared_mlp(feat_r)
        feat_s = self.shared_mlp(feat_s)

        # ------------- heads ------------- #
        rppg = self.ConvLast_hr(self.hr_head_norm(feat_r)).view(b, self.frames)
        spo2 = self.ConvLast_spo2(self.spo2_head_norm(feat_s)).view(b, self.frames)
        spo2 = rounding_sigmoid_approximation(100 * torch.sigmoid(spo2), k=10)
        return rppg, spo2,self.lambda_task
    
