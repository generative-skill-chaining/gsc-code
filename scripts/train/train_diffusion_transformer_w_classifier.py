#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Union
import pickle
import numpy as np
import torch
import tqdm
import os 

from generative_skill_chaining import agents, envs
from generative_skill_chaining.utils import random, tensors

from generative_skill_chaining.diff_models.unet_transformer import ScoreNet, ScoreNetState
from generative_skill_chaining.diff_models.classifier_transformer import ScoreModelMLP, TransitionModel
from generative_skill_chaining.mixed_diffusion.cond_diffusion1D import Diffusion
from generative_skill_chaining.mixed_diffusion.datasets_transformer_class import get_primitive_loader


@tensors.numpy_wrap
def query_observation_vector(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.encoder.encode(observation.to(policy.device), policy_args)


def train_model(
    diffusion_model_transition: Diffusion,
    diffusion_model_state: Diffusion,
    transition_model: TransitionModel,
    score_model_classifier: Optional[ScoreModelMLP],
    optimizer_transition: torch.optim.Optimizer,
    optimizer_state: torch.optim.Optimizer,
    transition_optimizer: torch.optim.Optimizer,
    optimizer_classifier: Optional[torch.optim.Optimizer],
    dataloader_transition: torch.utils.data.DataLoader,
    dataloader_state: torch.utils.data.DataLoader,
    dataloader_classifier: Optional[torch.utils.data.DataLoader],
    num_epochs: int = 1000,
    device: str = "auto",
    verbose: bool = True,
    path: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """Trains the model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if verbose:
        print("Training model on device:", device)

    diffusion_model_transition.to(device)
    diffusion_model_state.to(device)
    transition_model.to(device)
    if score_model_classifier is not None:
        score_model_classifier.to(device)

    criterion_transition = torch.nn.MSELoss()
    criterion_classifier = torch.nn.BCELoss()

    for epoch in tqdm.tqdm(range(num_epochs)):

        for batch in dataloader_transition:
            data = batch
            data = data[:, :-transition_model.state_dim]
            target = batch[:, -transition_model.state_dim:]
            data = data.to(device).float()
            target = target.to(device).float()
            predicted = transition_model(data)
            loss = criterion_transition(predicted, target)

            transition_optimizer.zero_grad()
            loss.backward()
            transition_optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} Transition loss: {loss.item():.4f}")

        if score_model_classifier is not None:
            for batch in dataloader_classifier:
                data = batch
                data = data[:, :-1]
                target = batch[:, -1:]
                data = data.to(device).float()
                target = target.to(device).float()
                predicted = score_model_classifier(data)
                loss = criterion_classifier(predicted, target)

                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_classifier.step()

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} Classifier loss: {loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            if path is not None:
                torch.save(transition_model.state_dict(), path / "transition_model.pt")
                if score_model_classifier is not None:
                    torch.save(score_model_classifier.state_dict(), path / "score_model_classifier.pt")

    if path is not None:
        torch.save(transition_model.state_dict(), path / "transition_model.pt")
        if score_model_classifier is not None:
            torch.save(score_model_classifier.state_dict(), path / "score_model_classifier.pt")


def train_diffusion(
    checkpoint: Union[str, pathlib.Path],
    env_config: Optional[Union[str, pathlib.Path]] = None,
    dataset_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    diffusion_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    train_classifier: bool = False,
    debug_results: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    num_episodes: int = 100,
    seed: Optional[int] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    num_samples: int = 16,
    gui: Optional[bool] = None,
    verbose: bool = True,
    device: str = "auto",
) -> None:
    """Evaluates the policy either by loading an episode from `debug_results` or
    generating `num_eval_episodes` episodes.
    """
    if path is None and debug_results is None:
        raise ValueError("Either path or load_results must be specified")
    if seed is not None:
        random.seed(seed)

    # Load env.
    env_kwargs: Dict[str, Any] = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    if env_config is None:
        # Try to load eval env.
        env_config = pathlib.Path(checkpoint).parent / "eval/env_config.yaml"
    try:
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None

    # Load policy.
    policy = agents.load(
        checkpoint=checkpoint, env=env, env_kwargs=env_kwargs, device=device
    )

    assert isinstance(policy, agents.RLAgent)
    policy.eval_mode()

    if env is None:
        env = policy.env

    observation_preprocessor = lambda obs, params: query_observation_vector(policy, obs, params)

    all_loader_datasets = get_primitive_loader(dataset_checkpoint, observation_preprocessor, modes=[
        "transition", "state", "classifier"
    ] if train_classifier else ["transition", "state"])

    dataloader_transition, dataset_transition = all_loader_datasets[0]
    dataloader_state, dataset_state = all_loader_datasets[1]
    if train_classifier:
        dataloader_classifier, dataset_classifier = all_loader_datasets[2]

    score_model_transition = ScoreNet(
        num_samples=num_samples,
        sample_dim=dataset_transition[0].shape[0],
        condition_dim=0
    )

    score_model_state = ScoreNetState(
        num_samples=num_samples,
        sample_dim=dataset_state[0].shape[0],
        condition_dim=0
    )

    transition_model = TransitionModel(
        sample_dim=dataset_transition[0].shape[0],
        state_dim=dataset_state[0].shape[0]-8,
        out_channels=512
    )

    if train_classifier:
        score_model_classifier = ScoreModelMLP(
            out_channels=512,
            state_dim=dataset_state[0].shape[0]-8,
            sample_dim=dataset_classifier[0].shape[0],
        )

    diffusion_model_transition = Diffusion(
        net=score_model_transition,
    )

    diffusion_model_state = Diffusion(
        net=score_model_state,
    )

    optimizer_transition = torch.optim.Adam(
        diffusion_model_transition.parameters(), lr=learning_rate
    )

    optimizer_state = torch.optim.Adam(
        diffusion_model_state.parameters(), lr=learning_rate
    )

    transition_optimizer = torch.optim.Adam(
        transition_model.parameters(), lr=learning_rate
    )

    if train_classifier:
        optimizer_classifier = torch.optim.Adam(
            score_model_classifier.parameters(), lr=learning_rate
        )

    if not os.path.exists(diffusion_checkpoint):
        os.makedirs(diffusion_checkpoint)

    train_model(
        diffusion_model_transition=diffusion_model_transition,
        diffusion_model_state=diffusion_model_state,
        transition_model=transition_model,
        score_model_classifier=score_model_classifier if train_classifier else None,
        optimizer_transition=optimizer_transition,
        optimizer_state=optimizer_state,
        transition_optimizer=transition_optimizer,
        optimizer_classifier=optimizer_classifier if train_classifier else None,
        dataloader_transition=dataloader_transition,
        dataloader_state=dataloader_state,
        dataloader_classifier=dataloader_classifier if train_classifier else None,
        num_epochs=num_epochs,
        device=device,
        verbose=verbose,
        path=pathlib.Path(diffusion_checkpoint),
    )


def main(args: argparse.Namespace) -> None:
    train_diffusion(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--dataset-checkpoint", help="Dataset checkpoint")
    parser.add_argument("--diffusion-checkpoint", help="Diffusion checkpoint")
    parser.add_argument("--train-classifier", action="store_true", help="Train classifier")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--debug-results", type=str, help="Path to results_i.npz file.")
    parser.add_argument("--path", help="Path for output plots")
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--device", default="auto", help="Torch device")
    args = parser.parse_args()

    main(args)
