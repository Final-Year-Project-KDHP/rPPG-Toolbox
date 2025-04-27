import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba
from torch.nn import functional as F

################################################################################
# Utility Functions
################################################################################

def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Smooth alternative to torch.exp to prevent overflow.
    Keeps x smoothly bounded by using tanh(x / 400) * 400 inside the exponent.
    """
    return torch.exp(torch.tanh(x / 400) * 400)

def rounding_sigmoid_approximation(x: torch.Tensor, k: float, n_max: int = 100) -> torch.Tensor:
    """
    Vectorized, differentiable sigmoid-based approximation of a rounding function,
    using a safe_exp to avoid overflow issues.

    This is included here to mirror your SpO2 code. 
    It is not strictly used in the forward pass below but can be called 
    if you want a differentiable rounding behavior.
    """
    # Convert x to float64 for more stable summations
    x = x.to(torch.float64)
    
    # Create a tensor of all integer n values in the range [-n_max, ..., n_max]
    n_values = torch.arange(-n_max, n_max + 1, dtype=torch.float64, device=x.device)  
    
    # Expand x and n_values for broadcasting
    x_expanded = x.unsqueeze(-1)              
    n_expanded = n_values.view(1, 1, -1)      

    exponent1 = -k * (x_expanded - n_expanded + 0.5)
    exponent2 = -k * (x_expanded - n_expanded - 0.5)
    
    # Safe exponent terms
    term1 = n_expanded / (1 + safe_exp(exponent1))
    term2 = n_expanded / (1 + safe_exp(exponent2))
    
    result = (term1 - term2).sum(dim=-1, keepdim=True).squeeze(-1)
    return result

################################################################################
# Building Blocks
################################################################################

class ChannelAttention3D(nn.Module):
    """
    Computes channel-wise attention for 3D feature maps.
    """
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class LateralConnection(nn.Module):
    """
    Fuses features from fast and slow paths with a convolution that 
    downsamples the fast path to match the slow path's resolution/channels.
    """
    def __init__(self, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),   
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path


class CDC_T(nn.Module):
    """
    Central-Difference Convolution in the temporal dimension (with factor theta).
    Used for capturing temporal variations (e.g., pulse signals).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # For temporal kernel size > 1
            C_out, C_in, t, k1, k2 = self.conv.weight.shape
            if t > 1:
                # Sum weights at t=0 and t=2 for central-diff
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + \
                              self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, 
                                    stride=self.conv.stride, padding=0, 
                                    dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
            else:
                return out_normal


class MambaLayer(nn.Module):
    """
    A single Mamba layer (based on the mamba-ssm library), with residual connections
    and a dropout path. Used to capture long-range temporal dependencies.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0.0
        
        self.mamba = Mamba(
            d_model=dim,   # Model dimension
            d_state=d_state,  
            d_conv=d_conv,    
            expand=expand,    
            bimamba=True,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        """
        Reshapes 3D input [B, dim, T, H, W] into [B, T*H*W, dim], 
        applies Mamba, then reshapes back.
        """
        B, C, t, H, W = x.shape
        assert C == self.dim, "Channel dimension must match Mamba dimension"
        
        n_tokens = t * H * W
        # Flatten: [B, dim, T, H, W] -> [B, n_tokens, dim]
        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)  # (B, n_tokens, C)

        # Mamba
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))

        # Reshape back
        out = x_out.transpose(1, 2).reshape(B, C, t, H, W)
        return out

    def forward(self, x):
        # Mamba often prefers float32 for stability
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out


def conv_block(in_channels, out_channels, kernel_size, stride, padding, 
               bn=True, activation='relu'):
    """
    Standard 3D Conv block with optional BN and activation.
    """
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


################################################################################
# Multi-Task Model
################################################################################

class PhysMambaMultiTask(nn.Module):
    """
    Multi-task model that shares the entire backbone and produces:
      - rPPG / Heart Rate signal (HR head)
      - SpO2 prediction (SpO2 head)

    The 'split' occurs after the final pooled feature map, 
    which is [B, 48, frames, 1, 1]. 
    Each task then has its own "head" layers.
    """
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super(PhysMambaMultiTask, self).__init__()
        # frames=60
        self.frames = frames  # number of frames
        # print(f"Frames: {self.frames}")
        # -------------------------
        # 1) Shared Backbone
        # -------------------------

        self.ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)

        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)  # slow path
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)  # fast path

        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Mamba+CDC blocks for slow and fast streams
        self.Block1_slow = self._build_block(64, theta)
        self.Block2_slow = self._build_block(64, theta)
        self.Block3_slow = self._build_block(64, theta)

        self.Block1_fast = self._build_block(32, theta)
        self.Block2_fast = self._build_block(32, theta)
        self.Block3_fast = self._build_block(32, theta)

        # Lateral Connections
        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        # Pooling 
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # Dropouts
        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))


        # -------------------------
        # Add a Shared MLP/FC Layer
        # -------------------------
        # Here, a 1x1 convolution is used as an MLP that processes the channel dimension.
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(48, 48, kernel_size=1, bias=False),  # Acts like a fully connected layer on the channels.
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        # -------------------------
        # 2) Task-Specific Heads
        # -------------------------
        # Normalization layers at the start of each head
        self.hr_head_norm = nn.BatchNorm3d(48)
        self.spo2_head_norm = nn.BatchNorm3d(48)

        # --- (A) rPPG / HR Head ---
        self.ConvLast_hr = nn.Conv3d(48, 1, [1, 1, 1])

        # --- (B) SpO2 Head ---
        self.ConvLast_spo2 = nn.Conv3d(48, 1, [1, 1, 1])
        self.fc_spo2 = nn.Linear(frames, 1)  # final linear layer for SpO2

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )

    def forward_backbone(self, x):
        """
        Shared feature extractor that processes input [B,3,T,H,W]
        and returns a feature map [B,48,frames,1,1].
        """
        # Initial conv blocks
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        # Split into slow/fast streams
        s_x = self.ConvBlock4(x)  # slow path
        f_x = self.ConvBlock5(x)  # fast path

        # 1st set of blocks
        s_x1 = self.Block1_slow(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block1_fast(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1, f_x1)  # lateral fusion

        # 2nd set of blocks
        s_x2 = self.Block2_slow(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)

        f_x2 = self.Block2_fast(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2, f_x2)

        # 3rd block & upsampling
        s_x3 = self.Block3_slow(s_x2)
        s_x3 = self.upsample1(s_x3)
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block3_fast(f_x2)
        f_x3 = self.ConvBlock6(f_x3)
        f_x3 = self.drop_6(f_x3)

        # Final fusion + upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1)  # [B, 32+64=96, T', 1, 1]
        x_final = self.upsample2(x_fusion)         # [B, 48, T'', 1, 1]

        # Pool over spatial dims -> [B, 48, frames, 1, 1]
        x_final = self.poolspa(x_final)
        # print("x_final shape before poolspa:", x_final.shape)

        return x_final

    def hr_head(self, x):
        """
        rPPG / HR head. 
        Expects x of shape [B, 48, frames, 1, 1]. 
        Produces rPPG of shape [B, frames].
        """
        # Normalize the input for the HR head
        x = self.hr_head_norm(x)

        
        x_hr = self.ConvLast_hr(x)        # [B, 1, frames, 1, 1]
        # print("x_hr shape before view:", x_hr.shape)

        rPPG = x_hr.view(-1, self.frames) # [B, frames]
        return rPPG

    def spo2_head(self, x):
        """
        SpO2 head.
        Expects x of shape [B, 48, frames, 1, 1].
        Produces an SpO2 scalar (or small vector) for each sample.
        """
        # Normalize the input for the SpO2 head
        x = self.spo2_head_norm(x)

        x_spo2 = self.ConvLast_spo2(x)           # [B, 1, frames, 1, 1]
        flat_spo2 = x_spo2.view(-1, self.frames) # [B, frames]

        # Final FC: We map [B, frames] -> [B, 1], then scale.
        out_pre = self.fc_spo2(flat_spo2)          # [B, 1]
        output_pre_round = 100.0 * torch.sigmoid(out_pre) # e.g. in [0, 100]

        # Rounding approximation
        spo2_pred = rounding_sigmoid_approximation(output_pre_round, k=10)

        return spo2_pred

    def forward(self, x):
        """
        Forward pass returning both rPPG (HR) and SpO2 predictions.
        """
        # 1) Shared backbone
        features = self.forward_backbone(x)  # [B, 48, frames, 1, 1]

        # 2) Apply the shared MLP/FC layer
        features = self.shared_mlp(features)        # Process features further
        # print("features shape before hr_head:", features.shape)


        # 3) Task-specific heads
        rppg = self.hr_head(features)      # [B, frames] 
        spo2 = self.spo2_head(features)    # [B, 1]

        return rppg, spo2
