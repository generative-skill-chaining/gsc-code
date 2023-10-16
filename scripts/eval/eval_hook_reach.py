#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import symbolic
import tqdm
import json
from PIL import Image

from generative_skill_chaining import agents, envs
from generative_skill_chaining.envs.pybullet.table import primitives as table_primitives
from generative_skill_chaining.utils import recording, timing, random, tensors

from generative_skill_chaining.diff_models.unet_transformer import ScoreNet, ScoreNetState
from generative_skill_chaining.diff_models.classifier_transformer import ScoreModelMLP, TransitionModel
from generative_skill_chaining.mixed_diffusion.cond_diffusion1D import Diffusion


@tensors.numpy_wrap
def query_observation_vector(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.encoder.encode(observation.to(policy.device), policy_args)

def transform_forward(
    observation: torch.Tensor,
    indices: np.ndarray,
) -> torch.Tensor:

    if len(observation.shape) == 1:
        curr_size = observation.shape[0]
        return observation.reshape(8, 12)[indices].reshape(curr_size)

    curr_size = observation.shape[1]
    
    return observation.reshape(-1, 8, 12)[:, indices].reshape(-1, curr_size)

def transform_backward(
    observation: torch.Tensor,
    indices: np.ndarray,
) -> torch.Tensor:

    if len(observation.shape) == 1:
        curr_size = observation.shape[0]
        return observation.reshape(8, 12)[indices].reshape(curr_size)

    curr_size = observation.shape[1]

    return observation.reshape(-1, 8, 12)[:, indices].reshape(-1, curr_size)

def get_action_from_multi_diffusion(
    policies: Sequence[agents.RLAgent],
    diffusion_models: Sequence[Diffusion],
    transition_models: Sequence[TransitionModel],
    classifiers: Sequence[ScoreModelMLP],
    obs0: torch.Tensor,
    action_skeleton: Sequence[envs.Primitive],
    use_transition_model: bool = True,
    num_samples: int = 40,
    num_objects: int = 4,
    end_index: int = 24,
    state_dim: int = 96,
    action_dim: int = 4,
    gamma: Sequence[float] = [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    device: torch.device = "auto"
) -> np.ndarray:

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    for policy, diffusion_model, transition_model, classifier in zip(policies, diffusion_models, transition_models, classifiers):
        policy.to(device)
        diffusion_model.to(device)
        transition_model.to(device)
        classifier.to(device)

    num_steps = 256
    sample_dim = 0
    for i in range(len(policies)):
        sample_dim += state_dim + action_dim
    sample_dim += state_dim

    indices_dms = []
    indices_sdms = []

    for i in range(len(policies)):
        indices_dms.append((i*(state_dim+action_dim), (i+1)*(state_dim+action_dim)+state_dim))
        indices_sdms.append((i*(state_dim+action_dim), (i)*(state_dim+action_dim)+state_dim))

    xt = torch.zeros(num_samples, sample_dim).to(device)

    all_observation_indices = []
    all_reverse_observation_indices = []

    for i in range(len(policies)):
        observation_indices = np.array(action_skeleton[i].get_policy_args()["observation_indices"])
        reverse_observation_indices = np.zeros_like(observation_indices)

        for j in range(len(observation_indices)):
            reverse_observation_indices[observation_indices[j]] = j

        all_observation_indices.append(observation_indices)
        all_reverse_observation_indices.append(reverse_observation_indices)

    obs0 = np.array(obs0)*2
    x0 = torch.Tensor(obs0).to(device)

    mod_x0 = transform_backward(x0, all_reverse_observation_indices[0])

    all_sdes, all_ones, all_obs_ind, all_reverse_obs_ind = [], [], [], []

    for i in range(len(policies)):
        obs_ind = torch.Tensor(all_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        reverse_obs_ind = torch.Tensor(all_reverse_observation_indices[i]).to(device).unsqueeze(0).repeat(num_samples, 1)
        sde, ones = diffusion_models[i].configure_sdes(num_steps=num_steps, x_T=xt[:, indices_dms[i][0]:indices_dms[i][1]], num_samples=num_samples)
        all_sdes.append(sde)
        all_ones.append(ones)
        all_obs_ind.append(obs_ind)
        all_reverse_obs_ind.append(reverse_obs_ind)

    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        total_epsilon = torch.zeros_like(xt)

        all_epsilons = []

        for i, sde, ones, indices_dm, indices_sdm, obs_ind, reverse_obs_ind, transition_model, observation_indices, reverse_observation_indices in zip(range(len(policies)), all_sdes, all_ones, indices_dms, indices_sdms, all_obs_ind, all_reverse_obs_ind, transition_models, all_observation_indices, all_reverse_observation_indices):

            with torch.no_grad():
                sample = xt[:, indices_dm[0]:indices_dm[1]].clone()
                sample[:, :state_dim] = transform_forward(sample[:, :state_dim], observation_indices)
                sample[:, -state_dim:] = transform_forward(sample[:, -state_dim:], observation_indices)

                epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, sample, obs_ind)
                
                pred_x0 = (sample - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)

                pred_x0[:, -state_dim+36:] = pred_x0[:, 36:state_dim]

            
            with torch.no_grad():
                if use_transition_model:
                    pred_x0[:, state_dim+action_dim:] = transition_model(torch.cat([pred_x0[:, :state_dim+action_dim], obs_ind], dim=1))

                pred_x0 = torch.clip(pred_x0, -1, 1)

                epsilon = (sample - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)

                epsilon[:, :state_dim] = transform_backward(epsilon[:, :state_dim], reverse_observation_indices)
                epsilon[:, -state_dim:] = transform_backward(epsilon[:, -state_dim:], reverse_observation_indices)

                total_epsilon[:, indices_dm[0]:indices_dm[1]] += epsilon

                all_epsilons.append(epsilon)

                if i > 0:
                    total_epsilon[:, indices_sdm[0]:indices_sdm[1]] = gamma[i]*all_epsilons[i-1][:, -state_dim:] + (1-gamma[i])*all_epsilons[i][:, :state_dim]

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*total_epsilon) / torch.sqrt(alpha_t)

        
        pred_x0[:, :state_dim] = mod_x0[:state_dim]

        for i in range(len(indices_sdms)):
            pred_x0[:, indices_sdms[i][0]+12:indices_sdms[i][0]+end_index] = mod_x0[12:end_index]
            pred_x0[:, indices_sdms[i][0]+12*num_objects:indices_sdms[i][0]+state_dim] = mod_x0[12*num_objects:state_dim]
            pred_x0[:, indices_sdms[i][0]+48:indices_sdms[i][0]+60] = mod_x0[48:60]

        pred_x0[:, -state_dim+12:-state_dim+end_index] = mod_x0[12:end_index]
        pred_x0[:, -state_dim+12*num_objects:] = mod_x0[12*num_objects:]

        pred_x0[:, -state_dim+48:-state_dim+60] = mod_x0[48:60]
            
        with torch.no_grad():

            new_epsilon = torch.randn_like(total_epsilon)

            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

    xt = xt.detach().cpu().numpy()

    all_scores = []

    for i in range(1, len(indices_sdms)):
        final_states = xt[:, indices_sdms[i][0]:indices_sdms[i][1]].copy()
        scores = classifiers[i-1](torch.cat([transform_forward(torch.Tensor(final_states).to(device), all_observation_indices[i-1]), all_obs_ind[i-1]], dim=1)).detach().cpu().numpy().squeeze()
        all_scores.append(scores)

    final_states = xt[:, -state_dim:].copy()
    scores = classifiers[-1](torch.cat([transform_forward(torch.Tensor(final_states).to(device), all_observation_indices[-1]), all_obs_ind[-1]], dim=1)).detach().cpu().numpy().squeeze()
    all_scores.append(scores)

    scores = np.array(all_scores).T

    assert scores.shape == (num_samples, len(policies))

    scores = np.prod(scores, axis=1)

    assert scores.shape == (num_samples,)

    print("scores:", scores)

    sorted_indices = np.argsort(scores)[::-1][:5]

    xt = xt[sorted_indices]

    all_states = []
    all_actions = []

    for i in range(len(policies)):
        all_states.append(xt[:, indices_sdms[i][0]:indices_sdms[i][1]]*0.5)
        all_actions.append(xt[:, indices_sdms[i][1]:indices_sdms[i][1]+action_dim])
    
    all_states.append(xt[:, -state_dim:]*0.5)

    return all_actions, all_states

def evaluate_episodes(
    env: envs.Env,
    skills: Sequence[str] = [],
    policies: Optional[Sequence[Optional[agents.RLAgent]]] = None,
    diffusion_models: Optional[Sequence[Optional[Diffusion]]] = None,
    transition_models: Optional[Sequence[Optional[TransitionModel]]] = None,
    classifier_models: Optional[Sequence[Optional[ScoreModelMLP]]] = None,
    observation_preprocessors: Optional[Sequence[Optional[Callable]]] = None,
    target_skill_sequence: Sequence[int] = [],
    target_length: int = 10,
    num_episodes: int = 5,
    path: Optional[Union[str, pathlib.Path]] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
    device: str = "auto",
) -> None:
    """Evaluates policy for the given number of episodes."""
    num_successes = 0
    pbar = tqdm.tqdm(
        range(num_episodes),
        desc=f"Evaluate {env.name}",
        dynamic_ncols=True,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    current_skill = skills[target_skill_sequence[0]]

    policy = policies[current_skill]
    diffusion_model_transition = diffusion_models[current_skill]
    transition_model = transition_models[current_skill]
    score_model_classifier = classifier_models[current_skill]
    obs_preprocessor = observation_preprocessors[current_skill]

    diffusion_model_transition.to(device)
    transition_model.to(device)
    score_model_classifier.to(device)

    all_results = []

    if not path.exists():
        path.mkdir(parents=True)

    # remove all gif files from path
    for f in path.iterdir():
        if f.suffix == ".gif" or f.suffix == ".png":
            f.unlink()

    all_rewards = None

    for ep in pbar:
        # Evaluate episode.
        observation, reset_info = env.reset() #seed=seed)
        print("reset_info:", reset_info, "env.task", env.task)
        seed = reset_info["seed"]
        initial_observation = observation
        observation_indices = reset_info["policy_args"]["observation_indices"]
        print("primitive:", env.get_primitive(), env.action_skeleton[0].get_policy_args()["observation_indices"])
        print("reset_info:", reset_info)

        rewards = []
        done = False

        obs0 = obs_preprocessor(observation, reset_info["policy_args"])
    
        target_skill_sequence = target_skill_sequence[:target_length]

        print("Considering skills:", skills[target_skill_sequence[0]], skills[target_skill_sequence[1]])

        actions, pred_states = get_action_from_multi_diffusion(
            policies=[policies[skills[i]] for i in target_skill_sequence],
            diffusion_models=[diffusion_models[skills[i]] for i in target_skill_sequence],
            transition_models=[transition_models[skills[i]] for i in target_skill_sequence],
            classifiers=[classifier_models[skills[i]] for i in target_skill_sequence],
            obs0=obs0,
            action_skeleton=env.action_skeleton,
            use_transition_model=False,
            num_objects=6,
            end_index=36,
            device=device
        )

        if verbose:
            print("observation:", observation_str(env, observation))
            print("observation tensor:", observation)

        print("actions:", actions)
        
        for j in range(actions[0].shape[0]):

            env.reset(seed=seed)
            env.set_observation(initial_observation)

            env.record_start()

            rewards = []

            for i, action in enumerate(actions):

                env.set_primitive(env.action_skeleton[i])

                try:
                    observation, reward, terminated, truncated, step_info = env.step(action[j])
                except Exception as e:
                    continue

                if verbose:
                    print("step_info:", step_info)
                    
                print(f"Action for: {skills[target_skill_sequence[i]]}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")

                rewards.append(reward)
                done = terminated or truncated

            success = np.prod(rewards) > 0

            env.record_stop()

            if success:
                env.record_save(path / f"eval_{ep}_{i}_{j}_success.gif", reset=True)

                imgs = []

                for state in pred_states:
                    curr_state = state[j].reshape(8, 12)
                    curr_state = policy.encoder.unnormalize(torch.Tensor(curr_state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                    env.set_observation(curr_state)
                    imgs.append(env.render())

                imgs = np.concatenate(imgs, axis=1)

                Image.fromarray(imgs).save(path / f"eval_{ep}_{i}_{j}_success.png")              

            else:
                env.record_save(path / f"eval_{ep}_{i}_{j}_fail.gif", reset=True)

                imgs = []

                for state in pred_states:
                    curr_state = state[j].reshape(8, 12)
                    curr_state = policy.encoder.unnormalize(torch.Tensor(curr_state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                    env.set_observation(curr_state)
                    imgs.append(env.render())

                imgs = np.concatenate(imgs, axis=1)

                Image.fromarray(imgs).save(path / f"eval_{ep}_{i}_{j}_fail.png")   

            if success:
                env.set_observation(initial_observation)
                break
            else:
                env.set_observation(initial_observation)

        rewards = np.array(rewards)

        if all_rewards is None:
            all_rewards = rewards
        else:
            all_rewards += rewards

        num_successes += int(success)
        pbar.set_postfix(
            {"rewards": all_rewards.tolist(), "successes": f"{num_successes} / {num_episodes}"}
        )

def evaluate_diffusion(
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    diffusion_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    closed_loop: int,
    pddl_domain: str,
    pddl_problem: str,
    max_depth: int = 5,
    timeout: float = 10.0,
    num_samples: int = 10,
    verbose: bool = False,
    load_path: Optional[Union[str, pathlib.Path]] = None,
    seed: Optional[int] = None,
    gui: Optional[int] = None,
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
        print("env_config:", env_config)
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None
        assert False

    all_skills = ["pick", "place", "pull", "push"]
    all_policies = {}
    all_diffusion_models = {}
    all_transition_models = {}
    all_classifier_models = {}
    all_observation_preprocessors = {}

    for i, policy_checkpoint, diffusion_checkpoint in zip(range(len(policy_checkpoints)), policy_checkpoints, diffusion_checkpoints):

        # Load policy.
        policy = agents.load(
            checkpoint=policy_checkpoint, env=env, env_kwargs=env_kwargs, device=device
        )

        assert isinstance(policy, agents.RLAgent)
        policy.eval_mode()

        observation_preprocessor = lambda obs, params: query_observation_vector(policy, obs, params)

        diffusion_checkpoint = pathlib.Path(diffusion_checkpoint)

        score_model_transition = ScoreNet(
            num_samples=num_samples,
            sample_dim=196,
            condition_dim=0
        )

        score_model_state = ScoreNetState(
            num_samples=num_samples,
            sample_dim=96,
            condition_dim=0
        )

        transition_model = TransitionModel(
            sample_dim=196,
            state_dim=96,
            out_channels=512
        )

        score_model_classifier = ScoreModelMLP(
            out_channels=512,
            state_dim=96,
            sample_dim=97,
        )

        diffusion_model_transition = Diffusion(
            net=score_model_transition,
        )

        # Load diffusion model.
        diffusion_model_transition.load_state_dict(torch.load(diffusion_checkpoint / "diffusion_model_transition.pt"))
        transition_model.load_state_dict(torch.load(diffusion_checkpoint / "transition_model.pt"))
        score_model_classifier.load_state_dict(torch.load(diffusion_checkpoint / "score_model_classifier.pt"))

        all_policies[all_skills[i]] = policy
        all_diffusion_models[all_skills[i]] = diffusion_model_transition
        all_transition_models[all_skills[i]] = transition_model
        all_classifier_models[all_skills[i]] = score_model_classifier
        all_observation_preprocessors[all_skills[i]] = observation_preprocessor

    target_skill_sequence = [0, 2, 1, 0, 1]
    target_length = len(target_skill_sequence)
    num_episodes = num_eval

    evaluate_episodes(
        env=env,
        skills=all_skills,
        policies=all_policies,
        diffusion_models=all_diffusion_models,
        transition_models=all_transition_models,
        classifier_models=all_classifier_models,
        observation_preprocessors=all_observation_preprocessors,
        target_skill_sequence=target_skill_sequence,
        target_length=target_length,
        num_episodes=num_episodes,
        path=pathlib.Path(path),
        verbose=verbose,
        seed=seed,
    )


def main(args: argparse.Namespace) -> None:
    evaluate_diffusion(**vars(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument(
        "--diffusion-checkpoints", "-c", nargs="+", help="Diffusion checkpoints"
    )
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument(
        "--closed-loop", default=1, type=int, help="Run closed-loop planning"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--pddl-domain", help="Pddl domain", default=None)
    parser.add_argument("--pddl-problem", help="Pddl problem", default=None)
    parser.add_argument(
        "--max-depth", type=int, default=4, help="Task planning search depth"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Task planning timeout"
    )
    args = parser.parse_args()

    main(args)
