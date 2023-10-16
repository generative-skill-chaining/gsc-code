#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from generative_skill_chaining.mixed_diffusion.utils.diff_utils import *
from generative_skill_chaining.diff_models.unet_transformer import ScoreNet
from generative_skill_chaining.mixed_diffusion.sde_cont import PluginReverseSDE


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class Diffusion(Module):

    def __init__(self, net:ScoreNet):
        super().__init__()
        self.net = net
        self.gensde = PluginReverseSDE(scorefunc=net)

    def get_loss(self, x_0, obs_ind=None):
        ###############################################
        # x_0:  Input sample, (B, N, D).
        # condition:  (B, C)
        ###############################################

        loss = self.gensde.dsm(x_0, obs_ind=obs_ind).mean()

        return loss

    @torch.no_grad()
    def sample(self, 
            device, 
            num_samples=16, 
            sample_dim=7, 
            num_steps=256, 
            return_diffusion=False, 
            grad_fn=None, 
            replace=None, 
            mask=None,
            dynamics=None):

        condition = None

        # if replace is not None:
        #     replace = replace.repeat(batch_size*int(num_samples/replace.size(0)), 1).to(condition.device)

        x_T = torch.randn([num_samples, sample_dim]).to(device)
        diffusion = {num_steps: x_T.cpu().numpy().reshape(num_samples, sample_dim)}

        sde = self.gensde
        delta = sde.T / num_steps
        sde.base_sde.dt = delta
        ts = torch.linspace(1, 0, num_steps + 1).to(x_T) * sde.T
        ones = torch.ones(num_samples, 1).to(x_T)/num_steps

        for t in range(num_steps, 0, -1):
            xt = sde.sample(ones * t, x_T, condition, grad_fn, replace=replace, mask=mask, dynamics=dynamics)

            # if replace is not None:
            #     xt[:, 0:replace.shape[1]] = replace

            x_T = xt

            diffusion[t - 1] = xt.cpu().numpy().reshape(num_samples, sample_dim)
        
        if return_diffusion:
            return diffusion[0], diffusion
        else:
            return diffusion[0]

    def configure_sdes(self, num_steps, x_T, num_samples=16):
        
        sde = self.gensde
        delta = sde.T / num_steps
        sde.base_sde.dt = delta
        ts = torch.linspace(1, 0, num_steps + 1).to(x_T) * sde.T
        ones = torch.ones(num_samples, 1).to(x_T)/num_steps

        return sde, ones

    def configure_sdes_forward(self, num_steps, x_0, num_samples=1):

        sde = self.gensde
        delta = sde.T / num_steps
        sde.base_sde.dt = delta
        ts = torch.linspace(1, 0, num_steps + 1).to(x_0) * sde.T
        ones = torch.ones(num_samples, 1).to(x_0)/num_steps

        return sde, ones