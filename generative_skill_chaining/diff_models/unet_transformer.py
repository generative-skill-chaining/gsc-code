# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class ObjectEmbedder(nn.Module):
    """
    Embeds object representations into vector representations.
    (N, C, O) -> (N, C, D)
    """
    def __init__(self, input_size, action_size, hidden_size, num_objects):
        super().__init__()

        self.action_size = action_size
        
        self.mlp_x1 = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        if self.action_size > 0:

            self.mlp_x2 = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

            self.mlp_a = nn.Sequential(
                nn.Linear(action_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 2*hidden_size, bias=True),
            )

        self.object_mlp = nn.Sequential(
            nn.Linear(num_objects, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x1, object_sequence, a1, x2):

        one_hot_object_sequence = F.one_hot(object_sequence.long(), num_classes=object_sequence.shape[1]).float()

        batch_size = x1.shape[0]
        x1 = self.mlp_x1(x1)

        if self.action_size > 0:
            x2 = self.mlp_x2(x2)
            a1 = self.mlp_a(a1.unsqueeze(1))

        y = self.object_mlp(one_hot_object_sequence)
        
        x1y = torch.cat([x1, y], dim=2)

        if self.action_size > 0:
            x2y = torch.cat([x2, y], dim=2)

            return torch.cat([x1y, a1, x2y], dim=1)
        else:
            return x1y


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        t = t.float().squeeze(-1)
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU() #approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # print(f'[ models/temporal ] DiTBlock input shape: {x.shape}')
        # print(f'[ models/temporal ] DiTBlock c shape: {c.shape} and {self.adaLN_modulation(c).chunk(6, dim=1)[0].shape}')
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, input_size, action_size, num_objects):
        super().__init__()
        self.num_objects = num_objects
        self.action_size = action_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_x1 = nn.Linear(hidden_size, input_size, bias=True)
        if self.action_size > 0:
            self.linear_x2 = nn.Linear(hidden_size, input_size, bias=True)
            self.linear_a1 = nn.Linear(hidden_size, action_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x1 = self.linear_x1(x[:, :self.num_objects])
        if self.action_size > 0:
            a1 = self.linear_a1(x[:, self.num_objects:self.num_objects+1])
            x2 = self.linear_x2(x[:, self.num_objects+1:])
            return x1, a1, x2
        else:
            return x1


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=12,
        action_size=4,
        num_objects=8,
        hidden_size=128,
        depth=10,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_objects = num_objects
        self.num_heads = num_heads
        self.action_size = action_size

        self.x_embedder = ObjectEmbedder(input_size, action_size, int(hidden_size/2), num_objects)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        if self.action_size > 0:
            self.pos_embed = nn.Parameter(torch.zeros(1, 2*num_objects+1, hidden_size), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_objects, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, input_size, action_size, num_objects)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((2*self.num_objects + 1) ** 0.5))
        if self.action_size > 0:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(2*self.num_objects + 1))
        else:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.num_objects))
        # print(f'[ models/temporal ] Positional embedding shape: {pos_embed.shape} and copy to {self.pos_embed.data.shape}')
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear_x1.weight, 0)
        nn.init.constant_(self.final_layer.linear_x1.bias, 0)
        if self.action_size > 0:
            nn.init.constant_(self.final_layer.linear_a1.weight, 0)
            nn.init.constant_(self.final_layer.linear_a1.bias, 0)
            nn.init.constant_(self.final_layer.linear_x2.weight, 0)
            nn.init.constant_(self.final_layer.linear_x2.bias, 0)


    def forward(self, x1, t, object_sequence, a1=None, x2=None):
        """
        Forward pass of DiT.
        x1: (N, C, O) tensor of spatial inputs (images or latent representations of images)
        x2: (N, C, O) tensor of spatial inputs (images or latent representations of images)
        a1: (N, A) tensor of action inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) object sequence labels
        """
        x = self.x_embedder(x1, object_sequence, a1, x2) 
        # print(f'[ models/temporal ] x_embedder output shape: {x.shape}')
        x = x + self.pos_embed                              # (N, C, O) + (1, C, O) -> (N, C, O)
        c = self.t_embedder(t)                              # (N, D)
        
        # print(f'[ models/temporal ] t_embedder output shape: {c.shape}')

        for block in self.blocks:
            x = block(x, c)                                 # (N, T, D)
        
        # print(f'[ models/temporal ] blocks output shape: {x.shape}')

        if self.action_size > 0:
            ex1, ea1, ex2 = self.final_layer(x, c)              # (N, T, C) -> (N, C, O), (N, A), (N, C, O)

            return ex1, ea1, ex2
        else:
            ex1 = self.final_layer(x, c)              # (N, T, C) -> (N, C, O), (N, A), (N, C, O)

            return ex1

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, odd_num=True, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size + 1, dtype=np.float32) if odd_num else np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size + 1]) if odd_num else grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def DiT_L_8(**kwargs):
    return DiT(depth=24, num_heads=16, **kwargs)

class ScoreNet(nn.Module):
    def __init__(
        self,
        num_samples=20,
        sample_dim=196,
        condition_dim=0,
        state_dim=96,
        action_dim=4,
        num_objects=8,
    ):

        super().__init__()

        self.num_samples = num_samples
        self.num_objects = num_objects
        self.sample_dim = sample_dim
        self.condition_dim = condition_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dit_model = DiT_L_8()

    def forward(self, x, time, object_sequence):
        '''
            x : [ batch x sample_dim ]
        '''

        object_sequence = object_sequence.to(x.device)

        if time.ndim > 1:
            time = time.squeeze(-1)
        elif time.ndim == 0:
            time = time.unsqueeze(0)

        batch_size = x.shape[0]

        x1 = x[:, :self.state_dim].reshape(batch_size, self.num_objects, -1)
        a1 = x[:, self.state_dim:self.state_dim+self.action_dim].reshape(batch_size, -1)
        x2 = x[:, self.state_dim+self.action_dim:].reshape(batch_size, self.num_objects, -1)

        # print(f'[ models/temporal ] x1 shape: {x1.shape}')
        # print(f'[ models/temporal ] a1 shape: {a1.shape}')
        # print(f'[ models/temporal ] x2 shape: {x2.shape}')

        ex1, ea1, ex2 = self.dit_model(x1, time, object_sequence, a1, x2)

        eps_x1 = ex1.reshape(batch_size, -1)
        eps_a1 = ea1.reshape(batch_size, -1)
        eps_x2 = ex2.reshape(batch_size, -1)

        return torch.cat([eps_x1, eps_a1, eps_x2], dim=1)

class ScoreNetState(nn.Module):
    def __init__(
        self,
        num_samples=20,
        sample_dim=196,
        condition_dim=0,
        state_dim=96,
        action_dim=4,
        num_objects=8,
    ):

        super().__init__()

        self.num_samples = num_samples
        self.num_objects = num_objects
        self.sample_dim = sample_dim
        self.condition_dim = condition_dim
        self.state_dim = state_dim

        self.dit_model = DiT_L_8(action_size=0)

    def forward(self, x, time, object_sequence):
        '''
            x : [ batch x sample_dim ]
        '''

        object_sequence = object_sequence.to(x.device)

        if time.ndim > 1:
            time = time.squeeze(-1)
        elif time.ndim == 0:
            time = time.unsqueeze(0)

        batch_size = x.shape[0]

        x1 = x[:, :self.state_dim].reshape(batch_size, self.num_objects, -1)
        # print(f'[ models/temporal ] x1 shape: {x1.shape}')
        # print(f'[ models/temporal ] a1 shape: {a1.shape}')
        # print(f'[ models/temporal ] x2 shape: {x2.shape}')

        ex1 = self.dit_model(x1, time, object_sequence)

        eps_x1 = ex1.reshape(batch_size, -1)

        return eps_x1

if __name__ == "__main__":

    object_sequence = np.array([0, 3, 1, 2, 4, 5, 6, 7])

    scorenet = ScoreNet(
        num_samples=100,
        sample_dim=196,
        condition_dim=0,
        state_dim=96,
        action_dim=4
    )

    x = torch.randn(10, 196)
    time = torch.randn(10, 1)

    x = torch.cat([x, torch.Tensor(object_sequence).unsqueeze(0).repeat(10, 1)], dim=1)
    
    eps = scorenet(x, time)

    print(eps.shape)