#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Union
import pickle
import json
import numpy as np
import torch
import tqdm
from PIL import Image
from generative_skill_chaining import agents, envs
from generative_skill_chaining.utils import random, tensors

from generative_skill_chaining.diff_models.unet_transformer import ScoreNet, ScoreNetState
from generative_skill_chaining.diff_models.classifier_transformer import ScoreModelMLP, TransitionModel
from generative_skill_chaining.mixed_diffusion.cond_diffusion1D import Diffusion

@tensors.numpy_wrap
def query_observation_vector(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.encoder.encode(observation.to(policy.device), policy_args)

def modify_gradient_classifier(
    samples: torch.Tensor,
    score_model: ScoreModelMLP,
    target: int,
    num_grad_steps: int = 50
):
    ############################################
    # Perform the last step discrimination (refinement)
    # candidate_samples: [batch_size, num_samples, sample_dim]
    # returns: [batch_size, num_samples, sample_dim]
    ############################################

    device = samples.device
    samples = samples.detach()

    prev_loss = 0

    for i in range(num_grad_steps):

        samples = samples.clone()
        samples.requires_grad = True

        predicted_score = score_model(samples)

        loss = torch.nn.BCELoss()(predicted_score, torch.ones_like(predicted_score)*target)

        loss = loss.mean()

        loss.backward()

        samples = samples - 0.05 * samples.grad - 0.02 * torch.randn_like(samples)

        samples = samples.detach()

        prev_loss = loss.item()        

    return samples

@torch.no_grad()
def diffusion_forward(
    diffusion_model: Diffusion,
    samples: np.ndarray,
    device: torch.device = "auto"
):

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_steps = 256

    samples = torch.Tensor(samples).to(device)
    diffusion_model = diffusion_model.to(device)

    sde, ones = diffusion_model.configure_sdes_forward(num_steps=num_steps, x_0=samples)

    all_samples = [samples.cpu().unsqueeze(1).numpy()]

    for t in range(num_steps):

        new_samples = diffusion_model.gensde.base_sde.sample(t*ones, samples)

        all_samples.append(new_samples.cpu().unsqueeze(1).numpy())

    all_samples = np.concatenate(all_samples, axis=0)

    return all_samples[:, 0, :][::-1][-50:]


def get_action_from_diffusion(
    policy,
    diffusion_model: Diffusion,
    transition_model: TransitionModel,
    classifier: ScoreModelMLP,
    obs0: torch.Tensor,
    observation_indices: np.ndarray,
    use_transition_model: bool = True,
    num_samples: int = 5,
    state_dim: int = 96,
    action_dim: int = 4,
    device: torch.device = "cpu"
) -> np.ndarray:
    """Samples an action from the diffusion model."""
    # Sample action from diffusion model.

    num_steps = 256
    sample_dim = state_dim + action_dim + state_dim

    xt = torch.zeros(num_samples, sample_dim).to(device)

    observation_indices = torch.Tensor(observation_indices).to(device).unsqueeze(0).repeat(num_samples, 1)

    sde, ones = diffusion_model.configure_sdes(num_steps=num_steps, x_T=xt, num_samples=num_samples)

    x0 = torch.Tensor(np.array(obs0)*2).to(device)
    
    for t in tqdm.tqdm(range(num_steps, 0, -1)):

        epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, xt, observation_indices)
        
        pred = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
        
        pred[:, :state_dim] = x0

        if use_transition_model:
            pred[:, state_dim+action_dim:] = transition_model(torch.cat([pred[:, :state_dim+action_dim], observation_indices], dim=1))

        pred = torch.clip(pred, -1, 1)

        epsilon = (xt - torch.sqrt(alpha_t)*pred) / torch.sqrt(1 - alpha_t)

        pred_x0 = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)

        pred_x0 = torch.clip(pred_x0, -1, 1)

        new_epsilon = torch.randn_like(epsilon)
        
        xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

    xt = xt.detach().cpu().numpy()

    initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    print("initial_state:", initial_state.shape, "observation_indices:", observation_indices.shape)

    scores = classifier(torch.cat([torch.Tensor(initial_state).to(device), observation_indices], dim=1)).detach().cpu().numpy().squeeze()

    print("scores:", scores)

    # arrange samples in descending order of scores
    sorted_indices = np.argsort(scores)[::-1]

    xt = xt[sorted_indices]

    import pickle

    initial_state, action, final_state = xt[:, :state_dim], xt[:, state_dim:state_dim+action_dim], xt[:, state_dim+action_dim:]

    return action, final_state*0.5, initial_state*0.5

def evaluate_episodes(
    env: envs.Env,
    policy: agents.RLAgent,
    diffusion_model_transition: Diffusion,
    transition_model: TransitionModel,
    score_model_classifier: ScoreModelMLP,
    obs_preprocessor: Any,
    num_episodes: int,
    path: pathlib.Path,
    verbose: bool,
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

    diffusion_model_transition.to(device)
    transition_model.to(device)
    score_model_classifier.to(device)

    all_results = []

    # remove all gif files from path
    for f in path.iterdir():
        if f.suffix == ".gif" or f.suffix == ".png":
            f.unlink()

    for i in pbar:
        # Evaluate episode.
        observation, reset_info = env.reset() #seed=seed)
        print("reset_info:", reset_info, "env.task", env.task)
        seed = reset_info["seed"]
        initial_observation = observation
        init_state = env.get_state()
        observation_indices = reset_info["policy_args"]["observation_indices"]
        if verbose:
            print("primitive:", env.get_primitive())
            print("reset_info:", reset_info)

        rewards = []
        done = False

        while not done:
            obs0 = obs_preprocessor(observation, reset_info["policy_args"])
            actions, pred_states, initial_states = get_action_from_diffusion(
                policy,
                diffusion_model_transition,
                transition_model,
                score_model_classifier,
                obs0,
                observation_indices=observation_indices,
                use_transition_model=False,
                device=device
            )

            if verbose:
                print("observation:", observation_str(env, observation))
                print("observation tensor:", observation)
                # print("action:", action_str(env, action))

            print("actions:", actions)

            for j, action in enumerate(actions):

                env.reset(seed=seed)
                env.set_observation(initial_observation)

                img_0 = env.render()

                env.record_start()

                try:
                    observation, reward, terminated, truncated, step_info = env.step(action)
                except Exception as e:
                    continue

                if verbose:
                    print("step_info:", step_info)
                    print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")

                rewards.append(reward)
                done = terminated or truncated

                diffusion_next_observation = policy.encoder.decode(torch.Tensor(pred_states[j]).clone().unsqueeze(0).to(device), reset_info["policy_args"]).cpu().numpy()
                diffusion_initial_observation = policy.encoder.decode(torch.Tensor(initial_states[j]).clone().unsqueeze(0).to(device), reset_info["policy_args"]).cpu().numpy()
                
                print("rewards:", rewards)
                print("done:", done)
                
                success = sum(rewards) > 0.0

                env.record_stop()

                if success:
                    env.record_save(path / f"eval_{i}_{j}_success.gif", reset=True)
                    true_next_observation = env.get_observation()
                    true_state = env.get_state()
                    true_obs0 = obs_preprocessor(true_next_observation, reset_info["policy_args"])
                    img = env.render()
                    img_true = np.array(img, dtype=np.uint8)
                    env.set_observation(diffusion_initial_observation)
                    img = env.render()
                    img_diffusion_initial = np.array(img, dtype=np.uint8)

                    env.set_observation(diffusion_next_observation)
                    img = env.render()
                    img_diffusion = np.array(img, dtype=np.uint8)
                    
                    Image.fromarray(img_true).save(path / f"eval_{i}_{j}_true.png")
                    
                    current_sample = np.concatenate([obs0, action, true_obs0])[None, :]

                    img_both = np.concatenate([img_diffusion_initial, img_diffusion], axis=1)
                    Image.fromarray(img_both).save(path / f"eval_{i}_{j}_pred.png")

                    all_diffusion_samples = diffusion_forward(
                        diffusion_model_transition,
                        current_sample
                    )

                    all_images_gif = []

                    for k in tqdm.tqdm(range(all_diffusion_samples.shape[0])):
                        
                        first_obs = policy.encoder.decode(torch.Tensor(all_diffusion_samples[k, :96]).unsqueeze(0).to(device), reset_info["policy_args"]).cpu().numpy()
                        second_obs = policy.encoder.decode(torch.Tensor(all_diffusion_samples[k, -96:]).unsqueeze(0).to(device), reset_info["policy_args"]).cpu().numpy()
                        action = all_diffusion_samples[k, 96:-96]

                        first_obs[1] = initial_observation[1]
                        second_obs[1] = initial_observation[1]

                        env.set_state(init_state)
                        env.set_observation(first_obs)
                        img = env.render()
                        img_first = np.array(img, dtype=np.uint8)

                        env.set_state(init_state)
                        env.set_observation(second_obs)
                        img = env.render()
                        img_second = np.array(img, dtype=np.uint8)

                        img_both = np.concatenate([img_first, img_second], axis=1)

                        all_images_gif.append(Image.fromarray(img_both))

                    for _ in range(50):
                        all_images_gif.append(Image.fromarray(img_both))

                    all_images_gif[0].save(path / f"eval_{i}_{j}_diffusion.gif", save_all=True, append_images=all_images_gif[1:], duration=100, loop=0)
                    all_images_gif[-1].save(path / f"eval_{i}_{j}_diffusion_last.png")

                    env.reset(seed=seed)

                    env.record_start()
                    
                    for k in tqdm.tqdm([0, 10, 20, 30, 40, 49]):
                        action = all_diffusion_samples[k, 96:100]
                        env.reset(seed=seed)
                        env.step(action)

                    env.record_stop()
                    env.record_save(path / f"eval_{i}_{j}_all_diffusion.gif", reset=True)

                else:
                    env.record_save(path / f"eval_{i}_{j}_fail.gif", reset=True)

                if success:
                    rewards = []
                    env.set_observation(initial_observation)
                    break
                else:
                    rewards = []
                    env.set_observation(initial_observation)

        num_successes += success
        pbar.set_postfix(
            {"rewards": rewards, "successes": f"{num_successes} / {num_episodes}"}
        )

    # save results as json
    with open(path / f"results_{seed}.json", "w") as f:
        json.dump(
            {
                "num_episodes": num_episodes,
                "num_successes": num_successes,
            }, f)



def evaluate_diffusion(
    checkpoint: Union[str, pathlib.Path],
    diffusion_checkpoint: Union[str, pathlib.Path],
    num_samples: int = 10,
    env_config: Optional[Union[str, pathlib.Path]] = None,
    debug_results: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    num_episodes: int = 1,
    seed: Optional[int] = None,
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
        print("env_config:", env_config)
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None
        assert False

    # Load policy.
    policy = agents.load(
        checkpoint=checkpoint, env=env, env_kwargs=env_kwargs, device=device
    )

    assert isinstance(policy, agents.RLAgent)
    policy.eval_mode()
    
    if env is None:
        env = policy.env

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

    evaluate_episodes(
        env=env,
        policy=policy,
        diffusion_model_transition=diffusion_model_transition,
        transition_model=transition_model,
        score_model_classifier=score_model_classifier,
        obs_preprocessor=observation_preprocessor,
        num_episodes=num_episodes,
        path=diffusion_checkpoint,
        verbose=verbose,
        seed=seed,
    )


def main(args: argparse.Namespace) -> None:
    evaluate_diffusion(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--diffusion-checkpoint", help="Diffusion checkpoint")
    # parser.add_argument("--env-config", help="Env config")
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
