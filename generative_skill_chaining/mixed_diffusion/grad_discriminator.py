#!/usr/bin/env python

import os
import random
import numpy as np
import torch
import torch.nn as nn

class GradDiscriminator():

    #################################################
    # Initialize the discriminator
    # score_models: list of score models
    # conditions: list of condition kwargs for each score model
    # batch_size: batch size
    #################################################

    def __init__(self, score_models, batch_size):
        self.score_models = score_models
        self.batch_size = batch_size

    def calc_grad(self, candidate_samples, conditions):

        ############################################
        # Calculate the gradient of the predicted scores 
        # with respect to the candidate samples
        # candidate_samples: [batch_size, num_samples, sample_dim]
        # returns: [batch_size, num_samples, sample_dim]
        ############################################

        device = candidate_samples.device

        candidate_samples = candidate_samples.reshape(self.batch_size, -1, candidate_samples.shape[-1])

        candidate_samples.requires_grad = True

        predicted_scores = []

        loss = 0
        for score_model, condition in zip(self.score_models, conditions):
            predicted_score = score_model(candidate_samples, **condition)
            loss += torch.abs(torch.ones_like(predicted_score) - predicted_score)**2

        loss = loss.mean()
        
        grad_value = torch.autograd.grad(loss, candidate_samples)[0] - 0.05 * torch.randn_like(candidate_samples)

        return grad_value.reshape(-1, grad_value.shape[-1]).to(device)

    def last_step_discrimination(self, candidate_samples, conditions, num_grad_steps=10):

        ############################################
        # Perform the last step discrimination (refinement)
        # candidate_samples: [batch_size, num_samples, sample_dim]
        # returns: [batch_size, num_samples, sample_dim]
        ############################################

        candidate_samples = torch.Tensor(candidate_samples)

        prev_loss = 0
        num_hits = 0

        candidate_samples.requires_grad = True

        while True:

            loss = 0

            for score_model, condition in zip(self.score_models, conditions):
                predicted_score = score_model(candidate_samples, **condition)
                loss += torch.abs(torch.ones_like(predicted_score) - predicted_score)**2

            loss = loss.mean()
            
            grad_value = torch.autograd.grad(loss, candidate_samples)[0] - 0.05 * torch.randn_like(candidate_samples)

            candidate_samples = candidate_samples - 0.01 * grad_value

            num_hits += 1

            if num_hits > num_grad_steps:
                break

        return candidate_samples

    def order_for_model(self, candidate_samples, score_model, condition, n_top):

        ############################################
        # Order the candidate samples for a specific score model
        # based on the predicted scores
        # candidate_samples: [batch_size, num_samples, sample_dim]
        # score_model: score model
        # condition: condition kwargs for the score model
        # n_top: number of top samples to return
        # returns: [batch_size, n_top, sample_dim]
        ############################################

        B, N, D = candidate_samples.shape # B: batch size, N: num samples, D: sample dimension

        predicted_scores = []

        predicted_scores = score_model(samples=torch.Tensor(candidate_samples), **condition)

        predicted_scores = predicted_scores.detach().cpu().numpy() # (B, N)

        indices_reo = np.argsort(predicted_scores, axis=1)[:, ::-1][:, :n_top] # (B, n_top)

        candidate_samples = candidate_samples[np.arange(B)[:, None], indices_reo] # (B, n_top, D)

        return candidate_samples

