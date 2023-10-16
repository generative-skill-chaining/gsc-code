#!/usr/bin/env python

##############################################################################
# Code derived from "Planning with Diffusion for Flexible Behavior Synthesis"
# Janner et al. (2022) https://diffusion-planning.github.io/
##############################################################################

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class Linear1DBlock(nn.Module):
    '''
        Linear --> Norm --> Mish
    '''

    def __init__(self, inp_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(inp_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class UpsampleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim):
        super(UpsampleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(embed_dim, dim_out, bias=False)
        self._hyper_gate = nn.Linear(embed_dim, dim_out)

    def forward(self, x, ctx):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret

class DownsampleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim):
        super(DownsampleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(embed_dim, dim_out, bias=False)
        self._hyper_gate = nn.Linear(embed_dim, dim_out)

    def forward(self, x, ctx):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret