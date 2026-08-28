from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import gym
import numpy as np
from reimplementation.src.gvpwm.examples import dino_oracle_utils
import torch
from omegaconf import OmegaConf

from reimplementation.src.gvpwm.adapters.dino_wm import DinoWorldModelAdapter
from reimplementation.src.gvpwm.config import PlannerConfig
from reimplementation.src.gvpwm.planner import GVPWMPlanner
from reimplementation.src.gvpwm.video import PrecomputedVideoPlanSource
from reimplementation.src.gvpwm.examples.dino_wall_oracle_utils import (
    candidate_wall_episodes,
    compute_wall_stats,
    load_wall_oracle_episode,
    make_wall_observation,
    make_wall_runtime_observation,
    replay_wall_episode_in_env,
    resolve_wall_data_dir,
    slice_wall_oracle_episode,
)
from extension.examples.wall_experiment_presets import (
    EXPERIMENT_CHOICES,
    EXPERIMENT_DESCRIPTIONS,
    apply_wall_experiment_preset,
)
from extension.feasibility2.integrate import load_feasibility_scorer


DINO_WM_ROOT = Path("/home/nvzutphen/dino_wm")

if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

import env as _dino_env  # noqa: F401  # registers "wall" with gym
from plan import load_model


DATA_ROOT = DINO_WM_ROOT / "data" / "wall_single"
MODEL_NAME = "wall_single"

DEFAULT_HORIZON = 25
DEFAULT_SPLIT = "all"
DEFAULT_NUM_EPISODES = 50
VERBOSE_DIAGNOSTICS = os.environ.get("WALL_VERBOSE_DIAGNOSTICS", "0") == "1"


def seed_episode(episode_idx: int, start_offset: int) -> None:
    seed = int(episode_idx * 100_000 + start_offset)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_episode_specs(value: str) -> list[tuple[int, int]]:
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            episode_text, offset_text = item.split(":", 1)
            specs.append((int(episode_text), int(offset_text)))
        else:
            specs.append((int(item), 0))
    return specs


def sample_wall_segment_specs(
    data_dir: Path,
    horizon: int,
    frame_skip: int,
    num_segments: int,
    seed: int,
    start_index: int = 0,
) -> list[tuple[int, int]]:
    states = torch.load(data_dir / "states.pth")
    required_frames = horizon * frame_skip + 1
    rng = random.Random(seed)
    sampled = []
    total_needed = start_index + num_segments
    while len(sampled) < total_needed:
        episode_idx = rng.randint(0, int(states.shape[0]) - 1)
        max_offset = int(states.shape[1]) - required_frames
        if max_offset < 0:
            continue
        sampled.append((episode_idx, rng.randint(0, max_offset)))
    return sampled[start_index:total_needed]


def to_runtime_observation(
    obs: dict,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return make_wall_runtime_observation(
        obs,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
    )


def step_env(
    env: gym.Env,
    action: torch.Tensor,
    action_repeat: int,
    primitive_action_dim: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
    env_action_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    actions = action.detach().reshape(action_repeat, primitive_action_dim)
    primitive_actions = actions * action_std.to(actions.device) + action_mean.to(actions.device)

    obs = None
    env_device = torch.device(getattr(env.unwrapped, "device", "cpu"))
    for primitive_action in primitive_actions:
        primitive_action = (
            primitive_action.to(device=env_device, dtype=torch.float32) * env_action_scale
        )
        step_out = env.step(primitive_action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
        else:
            obs, _, done, _ = step_out
            terminated, truncated = done, False
        if terminated or truncated:
            break

    if obs is None:
        raise RuntimeError("Wall environment did not return an observation.")

    if hasattr(env.unwrapped, "transform"):
        obs = dict(obs)
        obs["visual"] = env.unwrapped.transform(obs["visual"]).permute(1, 2, 0)

    return to_runtime_observation(obs, proprio_mean=proprio_mean, proprio_std=proprio_std)


def load_model_once(device: torch.device, model_name: str = MODEL_NAME):
    model_path = DINO_WM_ROOT / "checkpoints" / "outputs" / model_name
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    return model, model_cfg


def build_planner(
    world_model: DinoWorldModelAdapter,
    horizon: int,
    feasibility_checkpoint: str,
    experiment: str | None = None,
    disable_refinement: bool = False,
    inner_steps: int | None = None,
) -> GVPWMPlanner:
    config = PlannerConfig()
    config.mpc.horizon = horizon
    config.feasibility.checkpoint_path = feasibility_checkpoint
    if experiment is None:
        config.alm.dynamics_mode = "rollout"
        config.alm.use_dynamics_constraints = False
        config.alm.lambda_dynamics = 0.0
    else:
        apply_wall_experiment_preset(config, experiment)
    if disable_refinement:
        config.refinement.enabled = False
    if inner_steps is not None:
        config.alm.inner_steps = int(inner_steps)

    feasibility_model = None
    if config.feasibility.enabled:
        feasibility_model = load_feasibility_scorer(
            feasibility_checkpoint,
            device=world_model.device,
            freeze=True,
        )

    return GVPWMPlanner(
        world_model=world_model,
        config=config,
        feasibility_model=feasibility_model,
    )


def _oracle_initialize_latents_from_video(
    self,
    current_latent: torch.Tensor,
    video_latents: torch.Tensor,
) -> torch.Tensor:
    latents = video_latents.clone()
    latents[0] = current_latent
    return latents


def _oracle_video_alignment_loss(self, latent, reference):
    return super(DinoWorldModelAdapter, self).video_alignment_loss(latent, reference)


def _oracle_goal_loss(self, latent, goal_latent):
    return super(DinoWorldModelAdapter, self).goal_loss(latent, goal_latent)


def _prepare_wall_env(env: gym.Env, episode: dict, episode_idx: int) -> None:
    env.unwrapped.update_env(episode["env_info"])
    init_state = episode["states"][0].detach().cpu().numpy()
    env.unwrapped.seed(episode_idx)
    env.unwrapped.set_init_state(init_state)
    env.reset()


def _current_wall_state(env: gym.Env) -> np.ndarray:
    return env.unwrapped.dot_position.detach().cpu().numpy().astype(np.float32)


def evaluate_episode(
    episode_idx: int,
    start_offset: int,
    split: str,
    horizon: int,
    frame_skip: int,
    paper_horizon: int | None,
    data_dir: Path,
    stats: dict[str, torch.Tensor],
    feasibility_checkpoint: str,
    visual_only_guidance: bool = True,
    wm_history_length: int | None = None,
    allow_scale_mismatch: bool = False,
    wall_env_action_scale: float | None = None,
    wall_target_source: str = "env-replay",
    experiment: str | None = None,
    action_warm_start: str = "none",
    disable_refinement: bool = False,
    inner_steps: int | None = None,
    model=None,
    model_cfg=None,
    device=None,
) -> dict:
    print(
        f"[wall episode start] episode_idx={episode_idx} offset={start_offset} "
        f"split={split} horizon={horizon} frame_skip={frame_skip}"
    )
    if paper_horizon is None:
        paper_horizon = horizon * frame_skip
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or model_cfg is None:
        model, model_cfg = load_model_once(device)

    seed_episode(episode_idx, start_offset)

    episode = load_wall_oracle_episode(data_dir, episode_idx, stats=stats)
    if wall_target_source == "env-replay":
        target_env_action_scale = 1.0 if wall_env_action_scale is None else wall_env_action_scale
        episode = replay_wall_episode_in_env(
            episode,
            episode_idx=episode_idx,
            model_cfg=model_cfg,
            horizon=horizon,
            frame_skip=frame_skip,
            start_offset=start_offset,
            env_action_scale=target_env_action_scale,
        )
    elif wall_target_source == "dataset":
        target_env_action_scale = 0.5 if wall_env_action_scale is None else wall_env_action_scale
        episode = slice_wall_oracle_episode(
            episode,
            horizon=horizon,
            frame_skip=frame_skip,
            start_offset=start_offset,
        )
    else:
        raise ValueError(f"Unsupported wall_target_source={wall_target_source!r}")

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    _prepare_wall_env(env, episode, episode_idx)

    primitive_action_dim = int(getattr(env.unwrapped, "action_dim", 2))
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    if wm_action_dim % primitive_action_dim != 0:
        raise ValueError(
            f"World-model action dim {wm_action_dim} is not divisible by "
            f"Wall primitive action dim {primitive_action_dim}."
        )
    action_repeat = wm_action_dim // primitive_action_dim

    if VERBOSE_DIAGNOSTICS:
        print(
            f"[wall scale check] "
            f"primitive_action_dim={primitive_action_dim} "
            f"wm_action_dim={wm_action_dim} "
            f"action_repeat={action_repeat} "
            f"frame_skip={frame_skip} "
            f"horizon={horizon}"
        )
    if action_repeat != frame_skip:
        print(
            f"[wall scale WARNING] action_repeat ({action_repeat}) != frame_skip ({frame_skip}); "
            f"planner executes {action_repeat} primitive env steps per MPC step, "
            f"but oracle video advances {frame_skip} raw dataset steps per macro step."
        )
        if not allow_scale_mismatch:
            raise ValueError(
                "Invalid Wall time scale: action_repeat must match frame_skip for this checkpoint. "
                f"Use --frame-skip {action_repeat}, or pass --allow-scale-mismatch only for debugging."
            )

    action_mean = episode["action_mean"].to(device=device, dtype=torch.float32)
    action_std = episode["action_std"].to(device=device, dtype=torch.float32)
    action_low = episode["action_low"].to(device=device, dtype=torch.float32).repeat(action_repeat)
    action_high = episode["action_high"].to(device=device, dtype=torch.float32).repeat(action_repeat)

    if not visual_only_guidance:
        DinoWorldModelAdapter.initialize_latents_from_video = _oracle_initialize_latents_from_video
        DinoWorldModelAdapter.video_alignment_loss = _oracle_video_alignment_loss
        DinoWorldModelAdapter.goal_loss = _oracle_goal_loss

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
        planning_history_length=wm_history_length,
    )
    planner = build_planner(
        world_model=world_model,
        horizon=horizon,
        feasibility_checkpoint=feasibility_checkpoint,
        experiment=experiment,
        disable_refinement=disable_refinement,
        inner_steps=inner_steps,
    )

    print(
        f"[wall config] "
        f"experiment={experiment or 'default_rollout'} "
        f"dynamics_mode={planner.config.alm.dynamics_mode} "
        f"use_dynamics_constraints={planner.config.alm.use_dynamics_constraints} "
        f"lambda_dynamics={planner.config.alm.lambda_dynamics} "
        f"feasibility_enabled={planner.config.feasibility.enabled} "
        f"lambda_feasibility={planner.config.feasibility.lambda_feasibility} "
        f"lambda_transition={planner.config.feasibility.lambda_transition} "
        f"lambda_action_consistency={planner.config.feasibility.lambda_action_consistency}"
    )

    print(
        f"[wall planner] horizon={horizon} raw_horizon={paper_horizon} frame_skip={frame_skip} "
        f"inner={planner.config.alm.inner_steps} lr={planner.config.alm.learning_rate} "
        f"lambda_goal={planner.config.alm.lambda_goal} lambda_video={planner.config.alm.lambda_video} "
        f"lambda_action={planner.config.alm.lambda_action} "
        f"action_warm_start={action_warm_start} "
        f"refinement={'off' if not planner.config.refinement.enabled else f'{planner.config.refinement.num_samples}x{planner.config.refinement.noise_variance}:{planner.config.refinement.objective}'} "
        f"wm_history={world_model.history_length}/{world_model.model_history_length}"
    )

    trace_counter = {"step": 0, "primitive_steps": 0}
    goal_state = episode["states"][-1].detach().cpu().numpy()

    def expert_action_provider(time_index: int, remaining: int, device: torch.device):
        chunks = []
        for macro_index in range(time_index, time_index + remaining):
            raw0 = macro_index * frame_skip
            raw1 = raw0 + frame_skip
            if raw1 > episode["actions_normalized"].shape[0]:
                return None
            chunks.append(episode["actions_normalized"][raw0:raw1].reshape(-1))
        if not chunks:
            return None
        return torch.stack(chunks, dim=0).to(device=device, dtype=torch.float32)

    def traced_step_fn(action: torch.Tensor):
        t = trace_counter["step"]
        raw0 = t * frame_skip
        raw1 = raw0 + frame_skip

        action_matrix = action.detach().cpu().reshape(action_repeat, primitive_action_dim)
        planner_raw_actions = (action_matrix * action_std.cpu()) + action_mean.cpu()
        expert_raw_actions = episode["actions"][raw0:raw1].float()
        planner_macro_action = planner_raw_actions.sum(dim=0)
        planner_delta_est = planner_macro_action

        expert_macro_action = expert_raw_actions.sum(dim=0)
        expert_delta_est = expert_macro_action

        oracle_now = episode["states"][raw0]
        oracle_next = episode["states"][raw1]
        oracle_disp = oracle_next - oracle_now

        cur_before = _current_wall_state(env)
        to_goal_before = goal_state - cur_before
        planner_action_np = planner_macro_action.detach().cpu().numpy()
        expert_action_np = expert_macro_action.detach().cpu().numpy()

        planner_norm = float(np.linalg.norm(planner_action_np))
        goal_norm = float(np.linalg.norm(to_goal_before))
        expert_norm = float(np.linalg.norm(expert_action_np))

        planner_goal_dot = float(np.dot(planner_action_np, to_goal_before))
        expert_goal_dot = float(np.dot(expert_action_np, to_goal_before))

        planner_goal_cos = planner_goal_dot / (planner_norm * goal_norm + 1e-8)
        expert_goal_cos = expert_goal_dot / (expert_norm * goal_norm + 1e-8)
        planner_vs_expert_norm = planner_norm / (expert_norm + 1e-8)
        goal_dist_before = float(np.linalg.norm(to_goal_before))

        if VERBOSE_DIAGNOSTICS:
            print(
                f"[wall action trace {t}] "
                f"planner_action_sum=({planner_macro_action[0]:.3f},{planner_macro_action[1]:.3f}) "
                f"planner_delta_est=({planner_delta_est[0]:.2f},{planner_delta_est[1]:.2f}) "
                f"expert_action_sum=({expert_macro_action[0]:.3f},{expert_macro_action[1]:.3f}) "
                f"expert_delta_est=({expert_delta_est[0]:.2f},{expert_delta_est[1]:.2f}) "
                f"to_goal=({to_goal_before[0]:.2f},{to_goal_before[1]:.2f}) "
                f"goal_dist={np.linalg.norm(to_goal_before):.2f}"
            )
            print(
                f"[wall direction {t}] "
                f"planner_goal_dot={planner_goal_dot:.3f} "
                f"planner_goal_cos={planner_goal_cos:.3f} "
                f"expert_goal_dot={expert_goal_dot:.3f} "
                f"expert_goal_cos={expert_goal_cos:.3f} "
                f"planner_norm={planner_norm:.3f} "
                f"expert_norm={expert_norm:.3f} "
                f"norm_ratio={planner_vs_expert_norm:.3f}"
            )
        obs = step_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
            proprio_mean=episode["proprio_mean"],
            proprio_std=episode["proprio_std"],
            env_action_scale=target_env_action_scale,
        )

        cur_state = _current_wall_state(env)
        state_diff = np.linalg.norm(goal_state - cur_state)
        progress = goal_dist_before - float(state_diff)
        oracle_idx = min(t * frame_skip, episode["states"].shape[0] - 1)
        oracle_state = episode["states"][oracle_idx].detach().cpu().numpy()
        print(
            f"[wall env {t}] "
            f"progress={progress:.2f} "
            f"state_diff={state_diff:.2f} "
            f"planner_cos={planner_goal_cos:.3f} "
            f"expert_cos={expert_goal_cos:.3f} "
            f"norm_ratio={planner_vs_expert_norm:.3f} "
            f"planner_sum=({planner_macro_action[0]:.2f},{planner_macro_action[1]:.2f}) "
            f"expert_sum=({expert_macro_action[0]:.2f},{expert_macro_action[1]:.2f}) "
            f"state=({cur_state[0]:.2f},{cur_state[1]:.2f}) "
            f"goal=({goal_state[0]:.2f},{goal_state[1]:.2f}) "
            f"oracle_t=({oracle_state[0]:.2f},{oracle_state[1]:.2f})"
        )
        trace_counter["primitive_steps"] += action_repeat

        trace_counter["step"] += 1
        return obs

    if VERBOSE_DIAGNOSTICS:
        print("video_plan length:", len(episode["video_plan"]))
        print("expected macro video length:", horizon + 1)
        print("states length:", episode["states"].shape[0])
        print("actions length:", episode["actions"].shape[0])
        print("frame_skip:", frame_skip)
        print(
            f"wall layout: door={float(episode['env_info']['fix_door_location']):.2f} "
            f"wall={float(episode['env_info']['fix_wall_location']):.2f}"
        )

    debug_macro_indices = [0, 1, 2, 5, 10, 15, 20, horizon]
    debug_macro_indices = sorted({i for i in debug_macro_indices if i < len(episode["video_plan"])})
    if VERBOSE_DIAGNOSTICS:
        with torch.no_grad():
            debug_obs = [episode["video_plan"][i] for i in debug_macro_indices]
            encoded_dbg = world_model.encode_sequence(debug_obs).to(device)
            base = encoded_dbg[0]
            for j, macro_i in enumerate(debug_macro_indices):
                raw_i = macro_i * frame_skip
                st = episode["states"][raw_i]
                visual_loss_from_start = world_model.video_alignment_loss(base, encoded_dbg[j])
                state_dist_from_start = torch.linalg.norm(episode["states"][raw_i] - episode["states"][0])
                print(
                    f"[wall ref sensitivity] macro={macro_i} raw={raw_i} "
                    f"visual_loss_from_start={float(visual_loss_from_start.detach().cpu()):.6f} "
                    f"state_dist={float(state_dist_from_start.cpu()):.2f} "
                    f"state=({st[0]:.2f},{st[1]:.2f})"
                )

        with torch.no_grad():
            for k in range(min(3, horizon)):
                raw0 = k * frame_skip
                raw1 = raw0 + frame_skip
                expert_macro_actions = episode["actions"][raw0:raw1].float()
                expert_action_sum = expert_macro_actions.sum(dim=0)
                expert_delta_x1 = expert_action_sum
                expert_delta_x2 = expert_action_sum * 2.0
                oracle_delta = episode["states"][raw1] - episode["states"][raw0]
                print(
                    f"[wall expert scale {k}] "
                    f"raw_range=[{raw0}:{raw1}] "
                    f"expert_actions_shape={tuple(expert_macro_actions.shape)} "
                    f"expert_delta_x1=({expert_delta_x1[0]:.2f},{expert_delta_x1[1]:.2f}) "
                    f"expert_delta_x2=({expert_delta_x2[0]:.2f},{expert_delta_x2[1]:.2f}) "
                    f"oracle_delta=({oracle_delta[0]:.2f},{oracle_delta[1]:.2f})"
                )

    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=traced_step_fn,
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
        initial_warm_start_actions=(
            expert_action_provider(0, horizon, device)
            if action_warm_start == "expert"
            else None
        ),
        diagnostic_action_provider=expert_action_provider,
    )
    if VERBOSE_DIAGNOSTICS:
        print("Wall ALM planner finished")

    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = (first_macro * action_std.cpu()) + action_mean.cpu()
    expert_actions_norm = episode["actions_normalized"][:frame_skip]

    if VERBOSE_DIAGNOSTICS:
        print("first planner macro (normalized):", first_macro)
        print("first planner macro (raw env action scale):", first_macro_raw)
        print("first expert raw actions:", episode["actions"][:frame_skip])
        print("first expert normalized actions:", expert_actions_norm)

    cur_state = _current_wall_state(env)
    metrics = env.unwrapped.eval_state(goal_state, cur_state)

    if VERBOSE_DIAGNOSTICS:
        print("eval metrics:", metrics)
        print("agent/dot position:", cur_state)
        print("goal position:", goal_state)
        print("state_diff:", np.linalg.norm(goal_state - cur_state))
        print("success threshold: state_dist < 4.5")
    print(
        f"[wall ep {episode_idx}] split={split} horizon={horizon} "
        f"raw_horizon={paper_horizon} frame_skip={frame_skip} offset={start_offset} "
        f"success={metrics['success']} state_dist={metrics['state_dist']:.2f} "
        f"dyn_residual={result.steps[-1].dynamics_residual_norm:.4f}"
    )

    return {
        "episode_idx": episode_idx,
        "start_offset": int(start_offset),
        "split": split,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
        "executed_actions": int(result.executed_actions.shape[0]),
        "planning_horizon": int(horizon),
        "raw_horizon": int(paper_horizon),
        "frame_skip": int(frame_skip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GVP-WM Wall oracle evaluation with ALM latent collocation (rollout dynamics).")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Use 'all' for wall_single, or a folder split if present.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="World-model macro planning horizon")
    parser.add_argument("--raw-horizon", type=int, default=None, help="Paper/environment horizon T; overrides --horizon via T / frame_skip")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override model/data frame skip")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside the eligible episode list")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of eligible episodes to evaluate")
    parser.add_argument("--episode-ids", default=None, help="Comma-separated explicit episode ids to evaluate")
    parser.add_argument(
        "--episode-specs",
        default=None,
        help="Comma-separated episode[:offset] specs. Overrides --episode-ids when provided.",
    )
    parser.add_argument(
        "--sample-targets",
        action="store_true",
        help="Sample DINO-WM-style trajectory segments with replacement instead of using offset 0.",
    )
    parser.add_argument("--sample-seed", type=int, default=99, help="Seed for --sample-targets segment sampling")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Checkpoint folder under dino_wm/checkpoints/outputs")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Wall dataset directory")
    parser.add_argument(
        "--feasibility-checkpoint",
        type=str,
        default="/home/scur0196/DL2---Grounding-Generated-Videos-/checkpoints/transformer_sigma_delta.pt",
        help="Path to feasibility model checkpoint",
    )
    parser.add_argument(
        "--wm-history-length",
        type=int,
        default=None,
        help=(
            "Number of latent/action history frames to pass to DINO-WM during planning. "
            "Defaults to checkpoint num_hist; use 1 to match DINO-WM's one-frame "
            "planning rollout path."
        ),
    )
    parser.add_argument(
        "--visual-only-guidance",
        action="store_true",
        help="Deprecated compatibility flag; visual-only guidance is now the default.",
    )
    parser.add_argument(
        "--oracle-proprio-guidance",
        action="store_true",
        help="Diagnostic only: include oracle proprio latents in video/goal losses.",
    )
    parser.add_argument("--allow-scale-mismatch", action="store_true", help="Allow action_repeat != frame_skip for diagnostics only")
    parser.add_argument(
        "--wall-env-action-scale",
        type=float,
        default=None,
        help="Multiplier applied to denormalized Wall actions before env.step. Defaults to 1.0 for env-replay targets and 0.5 for dataset targets.",
    )
    parser.add_argument(
        "--wall-target-source",
        choices=("env-replay", "dataset"),
        default="env-replay",
        help="Use DINO-WM env-replayed dataset goals or raw dataset tensor goals.",
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_CHOICES,
        default=None,
        help="Optional experiment preset. Defaults to this script's original rollout configuration.",
    )
    parser.add_argument(
        "--action-warm-start",
        choices=("none", "expert"),
        default="none",
        help="Diagnostic only: initialize the first MPC solve from a supplied action sequence.",
    )
    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Diagnostic only: turn off random candidate refinement after gradient optimization.",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=None,
        help="Diagnostic only: override ALM/planner inner gradient steps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device, model_name=args.model_name)

    if args.experiment is not None:
        print(
            f"Using wall experiment preset: {args.experiment} - "
            f"{EXPERIMENT_DESCRIPTIONS[args.experiment]}"
        )

    frame_skip = args.frame_skip
    if frame_skip is None:
        frame_skip = int(getattr(model_cfg, "frameskip", 1))
    frame_skip = int(frame_skip)

    if args.raw_horizon is not None:
        if args.raw_horizon % frame_skip != 0:
            raise ValueError(
                f"--raw-horizon {args.raw_horizon} must be divisible by --frame-skip {frame_skip}."
            )
        macro_horizon = args.raw_horizon // frame_skip
        paper_horizon = args.raw_horizon
    else:
        macro_horizon = args.horizon
        paper_horizon = macro_horizon * frame_skip

    data_dir = resolve_wall_data_dir(args.data_root, args.split)
    stats = compute_wall_stats(data_dir)
    eligible = candidate_wall_episodes(data_dir, horizon=macro_horizon, frame_skip=frame_skip)
    if args.episode_specs:
        eval_specs = parse_episode_specs(args.episode_specs)
    elif args.sample_targets:
        eval_specs = sample_wall_segment_specs(
            data_dir,
            horizon=macro_horizon,
            frame_skip=frame_skip,
            num_segments=args.num_episodes,
            seed=args.sample_seed,
            start_index=args.start_index,
        )
    elif args.episode_ids:
        eval_specs = [(int(item), 0) for item in args.episode_ids.split(",") if item.strip()]
    else:
        eval_specs = [
            (idx, 0)
            for idx in eligible[args.start_index : args.start_index + args.num_episodes]
        ]

    mode_label = "Wall rollout" if args.experiment is None else f"Wall experiment={args.experiment}"
    print(
        f"Evaluating {mode_label} split={args.split} data_dir={data_dir} "
        f"macro_horizon={macro_horizon} raw_horizon={paper_horizon} frame_skip={frame_skip}"
    )
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode specs: {eval_specs}")

    results = []
    for idx, start_offset in eval_specs:
        print(f"\n=== Wall Episode {idx} offset {start_offset} ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    start_offset=start_offset,
                    split=args.split,
                    horizon=macro_horizon,
                    frame_skip=frame_skip,
                    paper_horizon=paper_horizon,
                    data_dir=data_dir,
                    stats=stats,
                    feasibility_checkpoint=args.feasibility_checkpoint,
                    visual_only_guidance=args.visual_only_guidance or not args.oracle_proprio_guidance,
                    wm_history_length=args.wm_history_length,
                    allow_scale_mismatch=args.allow_scale_mismatch,
                    wall_env_action_scale=args.wall_env_action_scale,
                    wall_target_source=args.wall_target_source,
                    experiment=args.experiment,
                    action_warm_start=args.action_warm_start,
                    disable_refinement=args.disable_refinement,
                    inner_steps=args.inner_steps,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                )
            )
        except Exception as exc:
            import traceback

            print(f"[wall ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Wall Evaluation Summary ===")
    for result in results:
        print(result)
    if results:
        success_rate = sum(result["success"] for result in results) / len(results)
        mean_dist = sum(result["state_dist"] for result in results) / len(results)
        mean_dyn = sum(result["dynamics_residual"] for result in results) / len(results)
        print(
            f"Success rate: {success_rate:.3f}  "
            f"mean_state_dist: {mean_dist:.2f}  "
            f"mean_dyn_res: {mean_dyn:.4f}"
        )


if __name__ == "__main__":
    main()
