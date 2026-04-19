from __future__ import annotations

import sys
from pathlib import Path

import gym
import torch
import numpy as np
from omegaconf import OmegaConf

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..losses import goal_mse
from ..video import temporal_resample_sequence
from .dino_oracle_utils import infer_horizon, load_oracle_episode


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD


DATA_DIR = DINO_WM_ROOT / "data" / "pusht_noise" / "train"
MODEL_NAME = "pusht"
MAX_HORIZON = None

# MPC hyperparameters
MPC_GRAD_STEPS = 50
MPC_LR = 0.05
MPC_LAMBDA_GOAL = 10.0
MPC_LAMBDA_VIDEO = 1.0
MPC_LAMBDA_ACTION = 0.05


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

    # inverse of dataset normalization only
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


def _optimize_actions(
    world_model: DinoWorldModelAdapter,
    latent_context: torch.Tensor,
    goal_latent: torch.Tensor,
    video_latents: torch.Tensor,
    horizon: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_steps: int,
    lr: float,
    lambda_goal: float,
    lambda_video: float,
    lambda_action: float,
    warm_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward-rollout gradient descent to optimize a sequence of actions."""
    device = world_model.device
    init = torch.zeros(horizon, world_model.action_dim, device=device)
    if warm_start is not None:
        n = min(warm_start.shape[0], horizon)
        init[:n] = warm_start[:n].to(device)

    actions_param = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([actions_param], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        bounded = torch.clamp(actions_param, action_low, action_high)

        z_hist = latent_context.clone()
        a_hist = torch.zeros(world_model.history_length, world_model.action_dim, device=device)
        rollout = [z_hist[-1]]
        for action in bounded:
            a_hist = torch.cat([a_hist, action.unsqueeze(0)], dim=0)[-world_model.history_length:]
            z_next = world_model.predict_next_latent(z_hist, a_hist)
            z_hist = torch.cat([z_hist, z_next.unsqueeze(0)], dim=0)[-world_model.history_length:]
            rollout.append(z_next)

        rollout_t = torch.stack(rollout, dim=0)

        loss = lambda_goal * goal_mse(rollout_t[-1], goal_latent)
        loss = loss + lambda_action * bounded.pow(2).sum()
        if lambda_video > 0.0:
            for t in range(1, rollout_t.shape[0] - 1):
                loss = loss + lambda_video * goal_mse(rollout_t[t], video_latents[t])

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            actions_param.clamp_(action_low, action_high)

    return actions_param.detach().clamp(action_low, action_high)


def run_mpc(
    world_model: DinoWorldModelAdapter,
    step_fn,
    start_obs,
    goal_obs,
    video_plan_frames,
    horizon: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_grad_steps: int = MPC_GRAD_STEPS,
    lr: float = MPC_LR,
    lambda_goal: float = MPC_LAMBDA_GOAL,
    lambda_video: float = MPC_LAMBDA_VIDEO,
    lambda_action: float = MPC_LAMBDA_ACTION,
) -> torch.Tensor:
    """MPC with forward-rollout gradient optimization at each step."""
    device = world_model.device

    # encode the full oracle video plan once
    encoded_video = world_model.encode_sequence(video_plan_frames).to(device)
    encoded_video = temporal_resample_sequence(encoded_video, horizon + 1)

    goal_latent = world_model.encode_observation(goal_obs).to(device)

    # initialize latent context (pad to history_length)
    start_latent = world_model.encode_observation(start_obs).to(device)
    latent_context = start_latent.unsqueeze(0).expand(world_model.history_length, -1, -1).clone()

    executed_actions = []
    warm_start = None
    time_index = 0

    while time_index < horizon:
        remaining = horizon - time_index
        current_video = temporal_resample_sequence(encoded_video[time_index:], remaining + 1)

        planned_actions = _optimize_actions(
            world_model=world_model,
            latent_context=latent_context,
            goal_latent=goal_latent,
            video_latents=current_video,
            horizon=remaining,
            action_low=action_low,
            action_high=action_high,
            num_steps=num_grad_steps,
            lr=lr,
            lambda_goal=lambda_goal,
            lambda_video=lambda_video,
            lambda_action=lambda_action,
            warm_start=warm_start,
        )

        action = planned_actions[0]
        next_obs = step_fn(action)
        next_latent = world_model.encode_observation(next_obs).to(device)
        executed_actions.append(action.clone())
        latent_context = torch.cat([latent_context, next_latent.unsqueeze(0)], dim=0)[-world_model.history_length:]

        # warm start: drop first action, append zero
        warm_start = torch.cat([
            planned_actions[1:],
            torch.zeros(1, world_model.action_dim, device=device),
        ], dim=0)

        time_index += 1
        if time_index % 5 == 0 or time_index == horizon:
            print(f"  [mpc step {time_index}/{horizon}] done")

    return torch.stack(executed_actions, dim=0)


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
        obs = reset_out[0]
    else:
        obs = reset_out

    velocities = torch.load("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/data/pusht_noise/train/velocities.pth")

    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = velocities[episode_idx, 0].detach().cpu().numpy()

    full_initial_state = np.concatenate([initial_state, initial_velocity], axis=0)
    env.unwrapped._set_state(full_initial_state)

    primitive_action_dim = int(env.action_space.shape[0])   # 2
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)  # 10
    action_repeat = wm_action_dim // primitive_action_dim   # 5

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

    print(f"starting MPC (horizon={horizon}, grad_steps={MPC_GRAD_STEPS})")
    executed_actions = run_mpc(
        world_model=world_model,
        step_fn=lambda action: step_env(
            env, action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        ),
        start_obs=episode["start_obs"],
        goal_obs=episode["goal_obs"],
        video_plan_frames=episode["video_plan"],
        horizon=horizon,
        action_low=action_low,
        action_high=action_high,
    )
    print("MPC finished")

    first_macro = executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
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

    print(f"[ep {episode_idx}] oracle_len={episode['length']} horizon={horizon} "
          f"success={metrics['success']} state_dist={metrics['state_dist']:.2f}")

    return {
        "episode_idx": episode_idx,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "executed_actions": int(executed_actions.shape[0]),
        "planning_horizon": int(horizon),
    }


EVAL_EPISODES = list(range(10))  # evaluate episodes 0-9


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    results = []
    for idx in EVAL_EPISODES:
        print(f"\n=== Episode {idx} ===")
        try:
            r = evaluate_episode(idx, model=model, model_cfg=model_cfg, device=device)
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"[ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Evaluation Summary ===")
    for r in results:
        print(r)
    if results:
        success_rate = sum(r["success"] for r in results) / len(results)
        mean_dist = sum(r["state_dist"] for r in results) / len(results)
        print(f"Success rate: {success_rate:.3f}  mean_state_dist: {mean_dist:.2f}")


if __name__ == "__main__":
    main()
