import torch 
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision

import os             # For file and directory handling
import cv2            # For video processing
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns # For better visualization
import tqdm          # For progress bars 0
import glob  

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, CenterCrop
from torchvision.transforms._transforms_video import ToTensorVideo

from addNoise import Diffusion


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        assert channels % heads == 0
        self.n_head = heads
        self.n_embd = channels  # burada n_embd = channels = C
        self.c_attn = nn.Linear(channels, 3 * channels)
        self.c_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, C, H, W] --> önce [B, H*W, C] formatına getir
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # -> [B, T=H*W, C]

        B, T, C = x.size()  # B, T, C = [32, 4096, 128] gibi olur

        qkv = self.c_attn(x)  # -> [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=2)  # her biri [B, T, C]

        # reshape to [B, nh, T, hs]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # CNN için causality gerekmez
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        # output proj + dropout
        y = self.dropout(self.c_proj(y))

        # tekrar orijinal boyuta getir: [B, C, H, W]
        y = y.transpose(1, 2).view(B, C, H, W)
        return y
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Init(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.convT=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True))



        
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.convT(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        return x + emb

class Up_last(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.convT=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2,bias=True),
            
            nn.GELU(approximate="tanh"))

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self,x,  t):
        x = self.convT(x)
        
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Generator(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = Init(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 4)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 4)
        self.down3 = Down(256, 512)
        self.sa3 = SelfAttention(512, 4)

        
        self.bot2 = DoubleConv(512, 512)
        
        self.up1 = Up(512, 256)
        self.sa4 = SelfAttention(256, 4)
        self.up2 = Up(256, 128)
        self.sa5 = SelfAttention(128, 4)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 4)
        self.u4 = Up_last(64, c_out)


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x,t)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        xb = self.bot2(x4)

        x = self.up1(xb, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.u4(x,t)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
 
