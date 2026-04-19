from __future__ import annotations

import sys
from pathlib import Path

import gym
import numpy as np
import torch
from omegaconf import OmegaConf

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..losses import goal_mse
from ..video import temporal_resample_sequence
from .dino_oracle_utils import infer_horizon, load_oracle_episode


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from plan import load_model


DATA_DIR = DINO_WM_ROOT / "data" / "pusht_noise" / "train"
MODEL_NAME = "pusht"
MAX_HORIZON = None

ACTION_ONLY_STEPS = 200
ACTION_ONLY_LR = 0.05
ACTION_ONLY_LAMBDA_GOAL = 10.0
ACTION_ONLY_LAMBDA_ACTION = 0.1
ACTION_ONLY_LAMBDA_VIDEO = 0.0

EVAL_EPISODES = list(range(1))


def to_runtime_observation(obs):
    return {
        "visual": torch.as_tensor(obs["visual"], dtype=torch.float32).permute(2, 0, 1) / 255.0,
        "proprio": torch.as_tensor(obs["proprio"], dtype=torch.float32)[..., :4],
    }


def step_env(
    env: gym.Env,
    action: torch.Tensor,
    action_repeat: int,
    primitive_action_dim: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    action_np = action.detach().cpu().numpy().reshape(action_repeat, primitive_action_dim)

    mean_np = action_mean.detach().cpu().numpy()
    std_np = action_std.detach().cpu().numpy()

    # Inverse of dataset normalization only; env applies action_scale/relative semantics.
    primitive_actions = (action_np * std_np) + mean_np

    obs = None
    for primitive_action in primitive_actions:
        step_out = env.step(primitive_action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
        else:
            obs, _, done, _ = step_out
            terminated, truncated = done, False

        if terminated or truncated:
            break

    return to_runtime_observation(obs)


def load_model_once(device):
    model_path = DINO_WM_ROOT / "checkpoints" / "outputs" / MODEL_NAME
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    return model, model_cfg


def optimize_actions_only(
    world_model: DinoWorldModelAdapter,
    observation_history,
    goal_observation,
    video_plan,
    horizon: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    lambda_goal: float,
    lambda_action: float,
    lambda_video: float,
    num_steps: int,
    lr: float,
) -> torch.Tensor:
    device = world_model.device

    latent_context = world_model.encode_sequence(observation_history).to(device)
    if latent_context.shape[0] < world_model.history_length:
        latent_context = latent_context[-1:].repeat(world_model.history_length, 1, 1)

    goal_latent = world_model.encode_observation(goal_observation).to(device)

    video_latents = None
    if lambda_video > 0.0:
        video_latents = world_model.encode_sequence(video_plan).to(device)
        video_latents = temporal_resample_sequence(video_latents, horizon + 1)

    actions = torch.nn.Parameter(torch.zeros(horizon, world_model.action_dim, device=device))
    optimizer = torch.optim.Adam([actions], lr=lr)

    bounded_actions = actions
    for step in range(num_steps):
        optimizer.zero_grad()

        bounded_actions = torch.clamp(actions, min=action_low, max=action_high)

        z_hist = latent_context.clone()
        a_hist = torch.zeros(
            world_model.history_length,
            world_model.action_dim,
            device=device,
        )
        pred_latents = [z_hist[-1]]
        for action in bounded_actions:
            a_hist = torch.cat([a_hist, action.unsqueeze(0)], dim=0)[-world_model.history_length:]
            z_next = world_model.predict_next_latent(z_hist, a_hist)
            z_hist = torch.cat([z_hist, z_next.unsqueeze(0)], dim=0)[-world_model.history_length:]
            pred_latents.append(z_next)

        pred_rollout = torch.stack(pred_latents, dim=0)

        goal_loss = goal_mse(pred_rollout[-1], goal_latent)
        action_loss = bounded_actions.pow(2).sum()

        video_loss = pred_rollout.new_tensor(0.0)
        if lambda_video > 0.0:
            for t in range(pred_rollout.shape[0]):
                video_loss = video_loss + goal_mse(pred_rollout[t], video_latents[t])

        loss = (
            lambda_goal * goal_loss
            + lambda_action * action_loss
            + lambda_video * video_loss
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            actions.clamp_(min=action_low, max=action_high)

        if step % 50 == 0 or step == num_steps - 1:
            print(
                f"[action-only step {step}] "
                f"goal={goal_loss.item():.4f} "
                f"video={video_loss.item():.4f} "
                f"action={action_loss.item():.4f}"
            )

    return bounded_actions.detach()




def evaluate_episode(episode_idx: int, model=None, model_cfg=None, device=None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or model_cfg is None:
        model, model_cfg = load_model_once(device)

    torch.manual_seed(episode_idx)

    episode = load_oracle_episode(DATA_DIR, episode_idx)
    frame_skip = 5
    horizon = infer_horizon(episode["length"], frame_skip=frame_skip, max_horizon=MAX_HORIZON)

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _ = reset_out[0]

    velocities = torch.load(DATA_DIR / "velocities.pth")
    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = velocities[episode_idx, 0].detach().cpu().numpy()
    full_initial_state = np.concatenate([initial_state, initial_velocity], axis=0)
    env.unwrapped._set_state(full_initial_state)

    primitive_action_dim = int(env.action_space.shape[0])
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    action_repeat = wm_action_dim // primitive_action_dim

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)

    rel_actions = torch.load(DATA_DIR / "rel_actions.pth").float()
    rel_actions = rel_actions / 100.0
    rel_actions = (rel_actions - ACTION_MEAN) / ACTION_STD
    primitive_low = rel_actions.amin(dim=(0, 1)).to(device=device, dtype=torch.float32)
    primitive_high = rel_actions.amax(dim=(0, 1)).to(device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )

    print("starting action-only optimization")
    optimized_actions = optimize_actions_only(
        world_model=world_model,
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        video_plan=episode["video_plan"],
        horizon=horizon,
        action_low=action_low,
        action_high=action_high,
        lambda_goal=ACTION_ONLY_LAMBDA_GOAL,
        lambda_action=ACTION_ONLY_LAMBDA_ACTION,
        lambda_video=ACTION_ONLY_LAMBDA_VIDEO,
        num_steps=ACTION_ONLY_STEPS,
        lr=ACTION_ONLY_LR,
    )
    print("action-only optimization finished")

    executed_actions = []
    for action in optimized_actions:
        _ = step_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        )
        executed_actions.append(action.detach().clone())

    executed_actions = torch.stack(executed_actions, dim=0)

    first_macro = executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0

    print("first action-only macro (normalized):", first_macro)
    print("first action-only macro (raw env scale):", first_macro_raw)
    print("first 5 expert actions:", episode["actions"][:5])

    expert_rel_norm = (episode["rel_actions"][:5].float() / 100.0 - action_mean.cpu()) / action_std.cpu()
    print("first 5 expert relative actions (normalized):", expert_rel_norm)

    goal_state = np.concatenate(
        [
            episode["states"][-1].detach().cpu().numpy(),
            episode["velocities"][-1].detach().cpu().numpy(),
        ],
        axis=0,
    )

    cur_state = np.array(
        [
            env.unwrapped.agent.position[0],
            env.unwrapped.agent.position[1],
            env.unwrapped.block.position[0],
            env.unwrapped.block.position[1],
            env.unwrapped.block.angle,
            env.unwrapped.agent.velocity[0],
            env.unwrapped.agent.velocity[1],
        ],
        dtype=np.float32,
    )

    metrics = env.unwrapped.eval_state(goal_state, cur_state)

    print("eval metrics:", metrics)
    print("agent position:", env.unwrapped.agent.position)
    print("agent velocity:", env.unwrapped.agent.velocity)
    print("block position:", env.unwrapped.block.position)
    print("block angle:", env.unwrapped.block.angle)

    print(
        f"[ep {episode_idx}] oracle_len={episode['length']} horizon={horizon} "
        f"success={metrics['success']} state_dist={metrics['state_dist']:.2f}"
    )

    return {
        "episode_idx": episode_idx,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "executed_actions": int(executed_actions.shape[0]),
        "planning_horizon": int(horizon),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    results = []
    for idx in EVAL_EPISODES:
        print(f"\n=== Episode {idx} ===")
        
        r = evaluate_episode(idx, model=model, model_cfg=model_cfg, device=device)
        results.append(r)

    print("\n=== Evaluation Summary ===")
    for r in results:
        print(r)
    if results:
        success_rate = sum(r["success"] for r in results) / len(results)
        mean_dist = sum(r["state_dist"] for r in results) / len(results)
        print(f"Success rate: {success_rate:.3f}  mean_state_dist: {mean_dist:.2f}")


if __name__ == "__main__":
    main()
