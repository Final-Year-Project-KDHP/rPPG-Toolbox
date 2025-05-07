""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
from torch.nn import functional as F

def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Smooth alternative to torch.exp to prevent overflow.
    Keeps x smoothly bounded by using tanh(x / 100) * 100 inside the exponent.
    """
    return torch.exp(torch.tanh(x / 400) * 400)

def rounding_sigmoid_approximation(x: torch.Tensor, k: float, n_max: int = 100) -> torch.Tensor:
    """
    Vectorized, differentiable sigmoid-based approximation of a rounding function,
    using a safe_exp to avoid overflow issues.
    
    Returns the same shape as the input.
    """
    # Convert x to float64 for more stable summations
    x = x.to(torch.float32)
    
    # Create a tensor of all integer n values in the range [-n_max, ..., n_max]
    # shape: [2 * n_max + 1]
    n_values = torch.arange(-n_max, n_max + 1, dtype=torch.float32, device=x.device)  
    
    # Expand x and n_values for broadcasting
    x_expanded = x.unsqueeze(-1)               # shape: [batch_size, 1, 1]
    n_expanded = n_values.view(1, 1, -1)       # shape: [1, 1, 2*n_max+1]
    
    exponent1 = -k * (x_expanded - n_expanded + 0.5)  # [batch_size, 1, 2*n_max+1]
    exponent2 = -k * (x_expanded - n_expanded - 0.5)  # [batch_size, 1, 2*n_max+1]
    
    # Compute "safe" exponent terms
    term1 = n_expanded / (1 + safe_exp(exponent1))    # [batch_size, 1, 2*n_max+1]
    term2 = n_expanded / (1 + safe_exp(exponent2))    # [batch_size, 1, 2*n_max+1]
    
    # Summation over the last dimension (the n-values dimension),
    # and squeeze the unnecessary dimension
    result = (term1 - term2).sum(dim=-1, keepdim=True).squeeze(-1)  # [batch_size, 1]
    
    return result


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=60):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.fc = nn.Linear(frames, 1)  # Reduces [B, 1, 160] to [B, 1]
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)
        rPPG = rPPG.squeeze(1)
        output = self.fc(rPPG)
        # output = 100 * torch.sigmoid(output)
        # output = rounding_sigmoid_approximation(output, 10)       # output = rounding_sigmoid_approximation(output, 1000)

        return output, x_visual, x_visual3232, x_visual1616
